import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardMSELoss(nn.Module):
    """标准均方误差损失"""
    def __init__(self):
        super(StandardMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


class CustomLoss(nn.Module):
    """
    复合多项式损失：
      L = λ_l2 * MSE + λ_poly * (|diff|^p + |diff|^q).sum()
    """
    def __init__(self, lambda_l2=1.0, lambda_poly=0.1, p=2, q=3):
        super(CustomLoss, self).__init__()
        self.lambda_l2   = lambda_l2
        self.lambda_poly = lambda_poly
        self.p = p
        self.q = q

    def forward(self, pred, target):
        diff = pred - target
        l2_loss   = torch.mean(diff ** 2)
        poly_loss = torch.sum(torch.abs(diff)**self.p + torch.abs(diff)**self.q)
        return self.lambda_l2 * l2_loss + self.lambda_poly * poly_loss


class CustomLossWithMask(nn.Module):
    """
    单一二值掩码加权 MSE：
      L = λ_l2 * MSE_global + λ_mask * mean((pred - target)^2 * mask)
    """
    def __init__(self, lambda_l2=1.0, lambda_mask=0.1):
        super(CustomLossWithMask, self).__init__()
        self.lambda_l2   = lambda_l2
        self.lambda_mask = lambda_mask

    def forward(self, pred, target, mask):
        l2_loss   = torch.mean((pred - target) ** 2)
        mask_loss = torch.mean((pred - target) ** 2 * mask)
        return self.lambda_l2 * l2_loss + self.lambda_mask * mask_loss


class WeightedMSELoss(nn.Module):
    """
    差异图加权 MSE：
      L = mean((pred - target)^2 * |pred - target|)
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target):
        mse_map        = (pred - target) ** 2
        difference_map = torch.abs(pred - target)
        weighted_map   = mse_map * difference_map
        return weighted_map.mean()


class SimpleLossWithMask(nn.Module):
    """
    简易二值掩码 MSE：
      L = mean((pred - target)^2 * mask)
    """
    def __init__(self):
        super(SimpleLossWithMask, self).__init__()

    def forward(self, pred, target, mask):
        return torch.mean((pred - target) ** 2 * mask)


class MultiModeWeightedMSELoss(nn.Module):
    """
    多模态加权 MSE Loss：
      L = MSE_global
        + sum_i (λ_i * mean((pred - target)^2 * weight_maps[:, i, :]))
    要求：
      - pred, target: Tensor(shape=[batch, output_dim])
      - weight_maps: Tensor(shape=[batch, num_modes, output_dim])
      - lambdas: list of float, len = num_modes
    """
    def __init__(self, lambdas):
        super(MultiModeWeightedMSELoss, self).__init__()
        self.lambdas = lambdas

    def forward(self, pred, target, masks, weight_maps):
        # 全局 MSE
        loss_global = F.mse_loss(pred, target)

        # 计算每元素平方误差并扩展到 [batch, 1, output_dim]
        diff_sq = (pred - target) ** 2
        diff_sq = diff_sq.unsqueeze(1)  # (batch, 1, output_dim)

        # 加权：按每个模态的权重图相乘
        # weight_maps: (batch, num_modes, output_dim)
        weighted = diff_sq * weight_maps  # (batch, num_modes, output_dim)

        # 对 output_dim、batch 两次求平均，得到每个模态的平均误差
        mode_means = weighted.mean(dim=2).mean(dim=0)  # (num_modes,)

        # 按 lambdas 加权求和
        loss_modes = sum(l * m for l, m in zip(self.lambdas, mode_means))

        return loss_global + loss_modes
