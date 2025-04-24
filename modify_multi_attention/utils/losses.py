import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target):
        # 计算均方误差损失
        mse_loss = (pred - target) ** 2

        # 计算差异图（可以是绝对误差或平方误差）
        difference_map = torch.abs(pred - target)  # 这里使用绝对误差作为示例

        # 用差异图加权损失
        weighted_loss = mse_loss * difference_map

        # 返回加权后的损失（平均损失）
        return weighted_loss.mean()


# utils/losses.py
import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, lambda_l2=1.0, lambda_poly=0.1, p=2, q=3):
        """
        :param lambda_l2: L2 损失的权重
        :param lambda_poly: 多项式损失的权重
        :param p: 差异项的平方惩罚（多项式的部分）
        :param q: 差异项的立方惩罚（多项式的部分）
        """
        super(CustomLoss, self).__init__()
        self.lambda_l2 = lambda_l2  # L2 损失的权重
        self.lambda_poly = lambda_poly  # 多项式损失的权重
        self.p = p  # 多项式平方惩罚
        self.q = q  # 多项式立方惩罚

    def forward(self, pred, target):
        """
        :param pred: 模型的预测值
        :param target: 真实值
        :return: 计算的损失值
        """
        # 计算标准的L2损失（均方误差损失）
        l2_loss = torch.mean((pred - target) ** 2)

        # 计算差异项
        difference = pred - target

        # 计算加和项：例如差异的平方 + 差异的立方
        poly_loss = torch.sum(torch.pow(torch.abs(difference), self.p) + torch.pow(torch.abs(difference), self.q))

        # 最终损失是L2损失和加和项的加权和
        total_loss = self.lambda_l2 * l2_loss + self.lambda_poly * poly_loss

        return total_loss
# utils/losses.py
import torch
import torch.nn as nn

class CustomLossWithMask(nn.Module):
    def __init__(self, lambda_l2=1.0, lambda_mask=0.1):
        super(CustomLossWithMask, self).__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_mask = lambda_mask

    def forward(self, pred, target, mask):
        # 计算 L2 损失
        l2_loss = torch.mean((pred - target) ** 2)

        # 计算加权损失
        weighted_loss = torch.mean((pred - target) ** 2 * mask)

        # 最终损失：L2 损失和加权损失的加权和
        total_loss = self.lambda_l2 * l2_loss + self.lambda_mask * weighted_loss
        return total_loss


class StandardMSELoss(nn.Module):
    def __init__(self):
        super(StandardMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)

class SimpleLossWithMask(nn.Module):
        def __init__(self):
            super(SimpleLossWithMask, self).__init__()

        def forward(self, pred, target, mask):
            # 计算标准的 L2 损失
            mse_loss = (pred - target) ** 2

            # 应用掩码来加权损失
            weighted_loss = mse_loss * mask

            # 返回加权后的损失（对每个样本求平均）
            return weighted_loss.mean()
