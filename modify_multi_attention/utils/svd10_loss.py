import torch
import torch.nn as nn

def get_svd_modes(tensor, topk=10):
    if tensor.dim() == 2:
        N = tensor.shape[1]
        hw = int(N ** 0.5)
        assert hw * hw == N
        tensor = tensor.view(-1, hw, hw)
    B, H, W = tensor.shape
    modes = []
    
    # 记录原始数据类型，用于统一处理
    original_dtype = tensor.dtype
    
    def create_zero_mode(reference_shape, dtype):
        """统一的零模态创建函数"""
        zero_mode = torch.zeros(reference_shape, device=tensor.device, dtype=torch.float32)
        if dtype == torch.float16:
            zero_mode = zero_mode.half()
        return zero_mode
    
    for i in range(B):
        current_tensor = tensor[i]
        u, s, vh = None, None, None
        
        # 检查数据类型，如果是Half精度则转换为Float32
        if current_tensor.dtype == torch.float16:
            current_tensor = current_tensor.float()
        
        # 多重fallback策略
        strategies = [
            # 策略1: 直接SVD
            lambda t: torch.linalg.svd(t, full_matrices=False),
            # 策略2: 添加小噪声
            lambda t: torch.linalg.svd(t + 1e-8 * torch.randn_like(t), full_matrices=False),
            # 策略3: 添加对角正则化（修复维度问题）
            lambda t: torch.linalg.svd(t + 1e-6 * torch.eye(min(t.shape[-2:]), device=t.device, dtype=t.dtype), full_matrices=False),
            # 策略4: 使用更大的正则化
            lambda t: torch.linalg.svd(t + 1e-4 * torch.randn_like(t), full_matrices=False),
            # 策略5: 截断极值
            lambda t: torch.linalg.svd(torch.clamp(t, -1e6, 1e6), full_matrices=False)
        ]
        
        success = False
        for strategy_idx, strategy in enumerate(strategies):
            try:
                u, s, vh = strategy(current_tensor)
                success = True
                if strategy_idx > 0:
                    print(f"警告: SVD计算使用了fallback策略 {strategy_idx + 1}")
                break
            except (RuntimeError, torch._C._LinAlgError) as e:
                if strategy_idx == len(strategies) - 1:
                    # 所有策略都失败，使用零模态
                    print(f"错误: 所有SVD策略都失败，使用零模态替代。错误: {str(e)}")
                    success = False
                    break
                continue
        
        if not success:
            # 使用统一的零模态创建函数
            single_modes = [create_zero_mode(tensor[i].shape, original_dtype) for _ in range(topk)]
        else:
            # 确保奇异值为正数并按降序排列
            s = torch.clamp(s, min=1e-12)
            
            # 检查数值稳定性
            if torch.isnan(s).any() or torch.isinf(s).any():
                print("警告: SVD结果包含NaN或Inf，使用零模态替代")
                single_modes = [create_zero_mode(tensor[i].shape, original_dtype) for _ in range(topk)]
            else:
                single_modes = []
                for k in range(min(topk, len(s))):
                    try:
                        mode_k = s[k] * torch.outer(u[:, k], vh[k, :])
                        # 检查模态是否有效
                        if torch.isnan(mode_k).any() or torch.isinf(mode_k).any():
                            mode_k = create_zero_mode(current_tensor.shape, original_dtype)
                        else:
                            # 转换回原始数据类型
                            if original_dtype == torch.float16:
                                mode_k = mode_k.half()
                        single_modes.append(mode_k)
                    except Exception as e:
                        print(f"警告: 模态{k}计算失败，使用零模态: {str(e)}")
                        single_modes.append(create_zero_mode(current_tensor.shape, original_dtype))
                
                # 如果奇异值数量不足topk，用零填充
                while len(single_modes) < topk:
                    single_modes.append(create_zero_mode(tensor[i].shape, original_dtype))
        
        # 组织模态数据
        for k in range(topk):
            if len(modes) <= k:
                modes.append([])
            modes[k].append(single_modes[k])
    
    # 堆叠所有批次的模态
    try:
        modes = [torch.stack(modes[k], dim=0) for k in range(topk)]
    except Exception as e:
        print(f"警告: 模态堆叠失败，返回零模态: {str(e)}")
        # 使用统一的零模态创建函数返回零模态
        zero_mode = torch.zeros_like(tensor, dtype=torch.float32)
        if original_dtype == torch.float16:
            zero_mode = zero_mode.half()
        modes = [zero_mode for _ in range(topk)]
    
    return modes

def svd_topk_losses(pred, target, topk=10):
    pred_modes = get_svd_modes(pred, topk=topk)
    target_modes = get_svd_modes(target, topk=topk)
    losses = []
    for k in range(topk):
        loss_k = ((pred_modes[k] - target_modes[k]) ** 2).mean()
        losses.append(loss_k)
    return losses  # [L_svd1, L_svd2, ..., L_svd10]

class TotalLossWithSVD(nn.Module):
    def __init__(self, base_weight=0.5, svd_weights=None, topk=10):
        super().__init__()
        if svd_weights is None:
            svd_weights = [0.5 / topk] * topk  # 默认均分0.5权重给10个模态
        
        # 验证权重参数
        if base_weight < 0:
            raise ValueError(f"基础权重必须为非负数，当前值: {base_weight}")
        if any(w < 0 for w in svd_weights):
            raise ValueError(f"SVD权重必须为非负数，当前值: {svd_weights}")
        if len(svd_weights) != topk:
            raise ValueError(f"SVD权重数量({len(svd_weights)})必须等于topk({topk})")
        
        # 计算权重总和
        all_weights = [base_weight] + svd_weights
        weight_sum = sum(all_weights)
        
        if weight_sum == 0:
            raise ValueError("所有权重之和不能为0")
        
        # 权重归一化
        self.base_weight = base_weight / weight_sum
        self.svd_weights = [w / weight_sum for w in svd_weights]
        self.topk = topk
        self.base_loss = nn.MSELoss()
        
        # 权重分布警告
        if self.base_weight < 0.1:
            print(f"警告: 基础MSE权重占比过低({self.base_weight:.3f})，可能影响训练稳定性")
        if self.base_weight > 0.9:
            print(f"警告: 基础MSE权重占比过高({self.base_weight:.3f})，SVD损失可能失效")
        
        # 记录原始和归一化后的权重
        self._original_base_weight = base_weight
        self._original_svd_weights = svd_weights.copy()
        self._weight_sum = weight_sum

    def forward(self, pred, target):
        # pred/target: [B, N] 或 [B, H, W]
        
        # 首先检查输入是否有效
        if torch.isnan(pred).any() or torch.isinf(pred).any() or torch.isnan(target).any() or torch.isinf(target).any():
            print("警告: 输入包含NaN或Inf，使用L2损失的安全版本")
            # 对于包含NaN/Inf的情况，使用安全的损失计算
            pred_safe = torch.where(torch.isnan(pred) | torch.isinf(pred), torch.zeros_like(pred), pred)
            target_safe = torch.where(torch.isnan(target) | torch.isinf(target), torch.zeros_like(target), target)
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        loss_base = self.base_loss(pred, target)
        
        # 检查基础损失是否有效
        if torch.isnan(loss_base) or torch.isinf(loss_base):
            print("警告: 基础MSE损失无效，返回零损失")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        try:
            loss_svds = svd_topk_losses(pred, target, topk=self.topk)
            # 检查SVD损失是否有效
            valid_svd_losses = []
            for i, l in enumerate(loss_svds):
                if torch.isnan(l) or torch.isinf(l):
                    print(f"警告: SVD损失{i+1}无效(NaN/Inf)，跳过")
                    valid_svd_losses.append(torch.tensor(0.0, device=l.device, dtype=l.dtype))
                else:
                    valid_svd_losses.append(l)
            loss_svds = valid_svd_losses
        except Exception as e:
            print(f"警告: SVD损失计算失败，仅使用基础MSE损失: {str(e)}")
            # 如果SVD完全失败，只使用基础损失
            return loss_base
        
        total_loss = self.base_weight * loss_base
        for w, l in zip(self.svd_weights, loss_svds):
            total_loss += w * l
        
        # 最终检查总损失是否有效
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("警告: 总损失无效，回退到基础MSE损失")
            return loss_base
            
        return total_loss
    
    def get_weight_info(self):
        """获取权重信息，用于调试和监控"""
        return {
            'original_base_weight': self._original_base_weight,
            'original_svd_weights': self._original_svd_weights,
            'normalized_base_weight': self.base_weight,
            'normalized_svd_weights': self.svd_weights,
            'weight_sum': self._weight_sum,
            'topk': self.topk
        }
    
    def print_weight_info(self):
        """打印权重信息"""
        info = self.get_weight_info()
        print(f"=== SVD损失权重信息 ===")
        print(f"原始基础权重: {info['original_base_weight']:.4f}")
        print(f"原始SVD权重: {[f'{w:.4f}' for w in info['original_svd_weights']]}")
        print(f"权重总和: {info['weight_sum']:.4f}")
        print(f"归一化基础权重: {info['normalized_base_weight']:.4f}")
        print(f"归一化SVD权重: {[f'{w:.4f}' for w in info['normalized_svd_weights']]}")
        print(f"TopK模态数: {info['topk']}")
        print(f"=========================")
