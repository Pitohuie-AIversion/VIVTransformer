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
    
    for i in range(B):
        success = False
        attempts = 0
        max_attempts = 3
        
        while not success and attempts < max_attempts:
            try:
                attempts += 1
                current_tensor = tensor[i]
                
                # 检查输入张量的数值稳定性
                if torch.isnan(current_tensor).any() or torch.isinf(current_tensor).any():
                    print(f"警告: 检测到NaN或Inf值，使用零张量替代 (batch {i}, attempt {attempts})")
                    current_tensor = torch.zeros_like(current_tensor)
                
                # 第一次尝试：直接SVD
                if attempts == 1:
                    u, s, vh = torch.linalg.svd(current_tensor, full_matrices=False)
                # 第二次尝试：添加正则化
                elif attempts == 2:
                    print(f"警告: SVD失败，尝试添加正则化 (batch {i})")
                    regularized_tensor = current_tensor + 1e-8 * torch.randn_like(current_tensor)
                    u, s, vh = torch.linalg.svd(regularized_tensor, full_matrices=False)
                # 第三次尝试：使用更强的正则化和条件数检查
                else:
                    print(f"警告: SVD再次失败，使用强正则化 (batch {i})")
                    # 添加对角正则化项
                    reg_strength = 1e-6
                    regularized_tensor = current_tensor + reg_strength * torch.eye(min(H, W), device=current_tensor.device, dtype=current_tensor.dtype)
                    u, s, vh = torch.linalg.svd(regularized_tensor, full_matrices=False)
                
                success = True
                
            except (RuntimeError, torch._C._LinAlgError) as e:
                if attempts >= max_attempts:
                    print(f"错误: SVD计算完全失败 (batch {i})，使用零模态: {str(e)}")
                    # 创建零模态作为备用方案
                    u = torch.zeros(H, min(H, W), device=tensor.device, dtype=tensor.dtype)
                    s = torch.zeros(min(H, W), device=tensor.device, dtype=tensor.dtype)
                    vh = torch.zeros(min(H, W), W, device=tensor.device, dtype=tensor.dtype)
                    success = True
                else:
                    print(f"警告: SVD失败 (batch {i}, attempt {attempts}): {str(e)}")
                    continue
        
        # 确保奇异值为正数并按降序排列
        s = torch.clamp(s, min=1e-12)
        
        single_modes = []
        for k in range(min(topk, len(s))):
            if k < len(s) and s[k] > 1e-12:  # 只使用有意义的奇异值
                mode_k = s[k] * torch.outer(u[:, k], vh[k, :])
            else:
                mode_k = torch.zeros_like(tensor[i])
            single_modes.append(mode_k)
        
        # 如果奇异值数量不足topk，用零填充
        while len(single_modes) < topk:
            zero_mode = torch.zeros_like(tensor[i])
            single_modes.append(zero_mode)
            
        for k in range(topk):
            if len(modes) <= k:
                modes.append([])
            modes[k].append(single_modes[k])
    
    modes = [torch.stack(modes[k], dim=0) for k in range(topk)]
    return modes

def svd_topk_losses(pred, target, topk=10):
    try:
        pred_modes = get_svd_modes(pred, topk=topk)
        target_modes = get_svd_modes(target, topk=topk)
        losses = []
        
        for k in range(topk):
            try:
                # 计算模态差异
                diff = pred_modes[k] - target_modes[k]
                
                # 检查数值稳定性
                if torch.isnan(diff).any() or torch.isinf(diff).any():
                    print(f"警告: 模态 {k+1} 计算出现NaN/Inf，使用零损失")
                    loss_k = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                else:
                    loss_k = (diff ** 2).mean()
                    
                    # 确保损失值是有限的
                    if torch.isnan(loss_k) or torch.isinf(loss_k):
                        print(f"警告: 模态 {k+1} 损失为NaN/Inf，使用零损失")
                        loss_k = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                
                losses.append(loss_k)
                
            except Exception as e:
                print(f"警告: 模态 {k+1} 损失计算失败: {str(e)}，使用零损失")
                loss_k = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                losses.append(loss_k)
        
        return losses  # [L_svd1, L_svd2, ..., L_svd10]
        
    except Exception as e:
        print(f"错误: SVD损失计算完全失败: {str(e)}，返回零损失")
        # 返回全零损失作为备用方案
        zero_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return [zero_loss] * topk

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
        try:
            loss_base = self.base_loss(pred, target)
            
            # 检查基础损失的有效性
            if torch.isnan(loss_base) or torch.isinf(loss_base):
                print("警告: 基础MSE损失为NaN/Inf，使用零损失")
                loss_base = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            
            try:
                loss_svds = svd_topk_losses(pred, target, topk=self.topk)
                total_loss = self.base_weight * loss_base
                
                # 添加SVD损失项
                for w, l in zip(self.svd_weights, loss_svds):
                    if torch.isnan(l) or torch.isinf(l):
                        print(f"警告: SVD损失项为NaN/Inf，跳过该项")
                        continue
                    total_loss += w * l
                
                # 最终检查总损失
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("警告: 总损失为NaN/Inf，回退到基础MSE损失")
                    return loss_base
                
                return total_loss
                
            except Exception as e:
                print(f"警告: SVD损失计算失败: {str(e)}，回退到基础MSE损失")
                return loss_base
                
        except Exception as e:
            print(f"错误: 损失计算完全失败: {str(e)}，返回零损失")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
    
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
