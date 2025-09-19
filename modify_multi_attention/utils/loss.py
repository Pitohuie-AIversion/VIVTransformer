import torch
import torch.nn as nn

def get_svd_modes(tensor, topk=3):
    if tensor.dim() == 2:
        N = tensor.shape[1]
        hw = int(N ** 0.5)
        assert hw * hw == N
        tensor = tensor.view(-1, hw, hw)
    B, H, W = tensor.shape
    modes = []
    for i in range(B):
        u, s, vh = torch.linalg.svd(tensor[i], full_matrices=False)
        single_modes = []
        for k in range(topk):
            mode_k = s[k] * torch.outer(u[:, k], vh[k, :])
            single_modes.append(mode_k)
        for k in range(topk):
            if len(modes) <= k:
                modes.append([])
            modes[k].append(single_modes[k])
    modes = [torch.stack(modes[k], dim=0) for k in range(topk)]
    return modes

def svd_topk_losses(pred, target, topk=3):
    pred_modes = get_svd_modes(pred, topk=topk)
    target_modes = get_svd_modes(target, topk=topk)
    losses = []
    for k in range(topk):
        loss_k = ((pred_modes[k] - target_modes[k]) ** 2).mean()
        losses.append(loss_k)
    return losses  # [L_svd1, L_svd2, L_svd3...]

class TotalLossWithSVD(nn.Module):
    def __init__(self, base_weight=0.5, svd_weights=None, topk=3):
        super().__init__()
        # 默认主损失占0.5，svd分配剩余0.5
        if svd_weights is None:
            svd_weights = [0.5/topk] * topk  # 比如[0.166,0.166,0.166]
        
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
        # pred/target: [B, N] or [B, H, W]
        loss_base = self.base_loss(pred, target)
        loss_svds = svd_topk_losses(pred, target, topk=self.topk)
        total_loss = self.base_weight * loss_base
        for w, l in zip(self.svd_weights, loss_svds):
            total_loss += w * l
        return total_loss

# 创建增强SVD损失的工厂函数
def create_enhanced_svd_loss(base_weight=0.5, svd_weights=None, topk=3, **kwargs):
    """
    创建增强SVD损失函数
    
    Args:
        base_weight: 基础损失权重
        svd_weights: SVD损失权重列表
        topk: SVD模式数量
        **kwargs: 其他参数（为了兼容性）
    
    Returns:
        TotalLossWithSVD实例
    """
    return TotalLossWithSVD(base_weight=base_weight, svd_weights=svd_weights, topk=topk)

def get_loss_function(loss_type: str = 'mse', **kwargs):
    """
    获取损失函数的统一接口
    
    Args:
        loss_type: 损失函数类型
            - 'mse': MSE损失
            - 'l1': L1损失
            - 'svd': SVD损失
            - 'mse_l1': MSE+L1组合损失
        **kwargs: 损失函数参数
    
    Returns:
        损失函数实例
    """
    if loss_type.lower() == 'mse':
        return nn.MSELoss()
    elif loss_type.lower() == 'l1':
        return nn.L1Loss()
    elif loss_type.lower() == 'svd':
        return create_enhanced_svd_loss(**kwargs)
    elif loss_type.lower() in ['mse_l1', 'combo']:
        # 尝试导入简化损失函数
        try:
            from .simplified_loss import create_simplified_loss
            return create_simplified_loss('mse_l1_combo', **kwargs)
        except ImportError:
            # 回退到MSE损失
            return nn.MSELoss()
    else:
        # 默认使用MSE损失
        return nn.MSELoss()
    
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
