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
# utils/losses.py

import torch
import torch.nn as nn

class SVDMainModeLoss(nn.Module):

    def __init__(self, h=200, w=200, mode='mse', lambda_main=1.0):
        super().__init__()
        self.h = h
        self.w = w
        self.mode = mode
        self.lambda_main = lambda_main

    def _extract_main_mode(self, tensor2d):
        U, S, Vh = torch.linalg.svd(tensor2d, full_matrices=False)
        main_mode = S[0] * torch.ger(U[:, 0], Vh[0, :])
        return main_mode, U[:, 0], Vh[0, :], S[0]

    def forward(self, pred, target):
        B = pred.shape[0]
        pred = pred.view(B, self.h, self.w)
        target = target.view(B, self.h, self.w)
        loss_sum = 0.0

        for i in range(B):
            pred_main, u_pred, v_pred, s_pred = self._extract_main_mode(pred[i])
            tgt_main, u_tgt, v_tgt, s_tgt = self._extract_main_mode(target[i])
            if self.mode == 'mse':
                loss = torch.mean((pred_main - tgt_main) ** 2)
            elif self.mode == 'l1':
                loss = torch.mean(torch.abs(pred_main - tgt_main))
            elif self.mode == 'energy':
                loss = (s_pred - s_tgt).abs()
            elif self.mode == 'cos':
                cos_loss = 1 - torch.abs(torch.dot(u_pred, u_tgt) / (u_pred.norm() * u_tgt.norm()))
                cos_loss += 1 - torch.abs(torch.dot(v_pred, v_tgt) / (v_pred.norm() * v_tgt.norm()))
                loss = cos_loss / 2
            else:
                raise ValueError("Unknown main mode loss type")
            loss_sum += loss
            if torch.isnan(loss_sum):
                print(f"[NaN in main mode loss] pred:{pred[i].max()}, tgt:{target[i].max()}")

            # print(f"[Batch SVDMainModeLoss] {loss_sum / B}")

        return self.lambda_main * loss_sum / B


class TotalLossWithSVD(nn.Module):
    """
    总loss = (lambda_base * 基础loss + lambda_main * SVD主模态loss) / (lambda_base + lambda_main)
    """
    def __init__(self, base_loss, svd_loss, lambda_base=1.0, lambda_main=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.svd_loss = svd_loss
        self.lambda_base = lambda_base
        self.lambda_main = lambda_main

    def forward(self, pred, target, mask):
        loss0 = self.base_loss(pred, target, mask)
        loss1 = self.svd_loss(pred, target)
        total = self.lambda_base * loss0 + self.lambda_main * loss1
        return total / (self.lambda_base + self.lambda_main)
import torch
import torch.nn as nn

class SVDTop3ModesLoss(nn.Module):
    def __init__(self, lambda_main=1.0):
        super().__init__()
        self.lambda_main = lambda_main

    def forward(self, pred, target, *args, **kwargs):  # 保留mask等多余参数兼容旧代码
        # pred, target: [B, T, H, W]
        diff = pred - target
        b, t, h, w = diff.shape
        total_main_energy = 0.0
        for bi in range(b):
            for ti in range(t):
                mat = diff[bi, ti]
                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                top3_energy = (S[:3] ** 2).sum()
                total_main_energy += top3_energy
        main_modes_loss = total_main_energy / (b * t)
        mse_loss = diff.pow(2).mean()
        total_loss = mse_loss + self.lambda_main * main_modes_loss
        return total_loss
