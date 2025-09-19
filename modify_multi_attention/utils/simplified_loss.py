import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MSEL1ComboLoss(nn.Module):
    """
    简化的MSE+L1组合损失函数
    相比复杂的SVD损失，这个损失函数更稳定且计算效率更高
    """
    
    def __init__(self, mse_weight: float = 0.8, l1_weight: float = 0.2, 
                 reduction: str = 'mean'):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.reduction = reduction
        
        # 确保权重和为1
        total_weight = mse_weight + l1_weight
        if total_weight != 1.0:
            logger.warning(f"权重和不为1 ({total_weight})，将进行归一化")
            self.mse_weight = mse_weight / total_weight
            self.l1_weight = l1_weight / total_weight
        
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.l1_loss = nn.L1Loss(reduction=reduction)
        
        logger.info(f"初始化MSE+L1组合损失: MSE权重={self.mse_weight:.3f}, L1权重={self.l1_weight:.3f}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测值 [B, ...]
            target: 目标值 [B, ...]
            
        Returns:
            组合损失值
        """
        # 确保输入形状匹配
        if pred.shape != target.shape:
            raise ValueError(f"预测值形状 {pred.shape} 与目标值形状 {target.shape} 不匹配")
        
        # 计算各项损失
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        
        # 组合损失
        total_loss = self.mse_weight * mse + self.l1_weight * l1
        
        return total_loss
    
    def get_component_losses(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回各组件损失的详细信息，用于监控
        """
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        total = self.mse_weight * mse + self.l1_weight * l1
        
        return {
            'mse': mse,
            'l1': l1,
            'total': total,
            'weighted_mse': self.mse_weight * mse,
            'weighted_l1': self.l1_weight * l1
        }

class AdaptiveMSEL1Loss(nn.Module):
    """
    自适应MSE+L1损失，根据训练进度动态调整权重
    在训练初期更依赖L1损失（更稳定），后期更依赖MSE损失（更精确）
    """
    
    def __init__(self, initial_mse_weight: float = 0.3, final_mse_weight: float = 0.8,
                 warmup_steps: int = 1000, reduction: str = 'mean'):
        super().__init__()
        self.initial_mse_weight = initial_mse_weight
        self.final_mse_weight = final_mse_weight
        self.warmup_steps = warmup_steps
        self.reduction = reduction
        self.step_count = 0
        
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.l1_loss = nn.L1Loss(reduction=reduction)
        
        logger.info(f"初始化自适应MSE+L1损失: 初始MSE权重={initial_mse_weight:.3f}, "
                   f"最终MSE权重={final_mse_weight:.3f}, 预热步数={warmup_steps}")
    
    def _get_current_weights(self) -> tuple:
        """根据当前步数计算权重"""
        if self.step_count >= self.warmup_steps:
            mse_weight = self.final_mse_weight
        else:
            # 线性插值
            progress = self.step_count / self.warmup_steps
            mse_weight = self.initial_mse_weight + progress * (self.final_mse_weight - self.initial_mse_weight)
        
        l1_weight = 1.0 - mse_weight
        return mse_weight, l1_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        
        # 获取当前权重
        mse_weight, l1_weight = self._get_current_weights()
        
        # 计算损失
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        
        total_loss = mse_weight * mse + l1_weight * l1
        
        return total_loss
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重信息"""
        mse_weight, l1_weight = self._get_current_weights()
        return {
            'mse_weight': mse_weight,
            'l1_weight': l1_weight,
            'step_count': self.step_count,
            'progress': min(1.0, self.step_count / self.warmup_steps)
        }

class PhysicsInformedLoss(nn.Module):
    """
    物理信息损失函数，在基础损失上添加物理约束
    适用于流体重建任务
    """
    
    def __init__(self, base_weight: float = 0.8, continuity_weight: float = 0.1,
                 smoothness_weight: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.base_weight = base_weight
        self.continuity_weight = continuity_weight
        self.smoothness_weight = smoothness_weight
        self.reduction = reduction
        
        # 基础损失
        self.base_loss = MSEL1ComboLoss(reduction=reduction)
        
        logger.info(f"初始化物理信息损失: 基础权重={base_weight:.3f}, "
                   f"连续性权重={continuity_weight:.3f}, 平滑性权重={smoothness_weight:.3f}")
    
    def _compute_continuity_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        计算连续性约束损失（散度应接近零）
        假设输入为 [B, H, W, 2] 格式，最后一维为 [u, v] 速度分量
        """
        if pred.dim() != 4 or pred.size(-1) != 2:
            # 如果不是速度场格式，返回零损失
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        u = pred[..., 0]  # [B, H, W]
        v = pred[..., 1]  # [B, H, W]
        
        # 计算梯度（使用有限差分）
        du_dx = u[..., :, 1:] - u[..., :, :-1]  # [B, H, W-1]
        dv_dy = v[..., 1:, :] - v[..., :-1, :]  # [B, H-1, W]
        
        # 调整尺寸以匹配
        min_h = min(du_dx.size(-2), dv_dy.size(-2))
        min_w = min(du_dx.size(-1), dv_dy.size(-1))
        
        du_dx = du_dx[..., :min_h, :min_w]
        dv_dy = dv_dy[..., :min_h, :min_w]
        
        # 散度
        divergence = du_dx + dv_dy
        
        # 连续性损失（散度的L2范数）
        if self.reduction == 'mean':
            return torch.mean(divergence ** 2)
        elif self.reduction == 'sum':
            return torch.sum(divergence ** 2)
        else:
            return divergence ** 2
    
    def _compute_smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        计算平滑性约束损失（总变分正则化）
        """
        if pred.dim() < 3:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # 计算梯度
        if pred.dim() == 4:  # [B, H, W, C]
            grad_x = pred[..., 1:, :, :] - pred[..., :-1, :, :]
            grad_y = pred[..., :, 1:, :] - pred[..., :, :-1, :]
        else:  # [B, H, W]
            grad_x = pred[..., 1:, :] - pred[..., :-1, :]
            grad_y = pred[..., :, 1:] - pred[..., :, :-1]
        
        # 总变分
        tv_loss = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
        
        return tv_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 基础损失
        base_loss = self.base_loss(pred, target)
        
        # 物理约束损失
        continuity_loss = self._compute_continuity_loss(pred)
        smoothness_loss = self._compute_smoothness_loss(pred)
        
        # 组合损失
        total_loss = (self.base_weight * base_loss + 
                     self.continuity_weight * continuity_loss +
                     self.smoothness_weight * smoothness_loss)
        
        return total_loss
    
    def get_component_losses(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取各组件损失详情"""
        base_loss = self.base_loss(pred, target)
        continuity_loss = self._compute_continuity_loss(pred)
        smoothness_loss = self._compute_smoothness_loss(pred)
        total_loss = (self.base_weight * base_loss + 
                     self.continuity_weight * continuity_loss +
                     self.smoothness_weight * smoothness_loss)
        
        return {
            'base_loss': base_loss,
            'continuity_loss': continuity_loss,
            'smoothness_loss': smoothness_loss,
            'total_loss': total_loss,
            'weighted_base': self.base_weight * base_loss,
            'weighted_continuity': self.continuity_weight * continuity_loss,
            'weighted_smoothness': self.smoothness_weight * smoothness_loss
        }

def create_simplified_loss(loss_type: str = "mse_l1_combo", **kwargs) -> nn.Module:
    """
    创建简化损失函数的工厂函数
    
    Args:
        loss_type: 损失函数类型
            - "mse_l1_combo": MSE+L1组合损失
            - "adaptive_mse_l1": 自适应MSE+L1损失
            - "physics_informed": 物理信息损失
        **kwargs: 损失函数参数
    
    Returns:
        损失函数实例
    """
    if loss_type == "mse_l1_combo":
        return MSEL1ComboLoss(**kwargs)
    elif loss_type == "adaptive_mse_l1":
        return AdaptiveMSEL1Loss(**kwargs)
    elif loss_type == "physics_informed":
        return PhysicsInformedLoss(**kwargs)
    else:
        logger.warning(f"未知损失类型 {loss_type}，使用默认MSE+L1组合损失")
        return MSEL1ComboLoss(**kwargs)

# 为了兼容性，提供与原有接口相同的函数
def create_enhanced_svd_loss_simplified(**kwargs):
    """
    创建简化版本的"增强SVD损失"（实际上是MSE+L1组合损失）
    用于替换复杂的SVD损失，提供更好的稳定性
    """
    logger.info("使用简化的MSE+L1损失替代复杂的SVD损失")
    return create_simplified_loss("mse_l1_combo", **kwargs)