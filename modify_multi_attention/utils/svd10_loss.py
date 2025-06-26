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
        all_weights = [base_weight] + svd_weights
        weight_sum = sum(all_weights)
        self.base_weight = base_weight / weight_sum
        self.svd_weights = [w / weight_sum for w in svd_weights]
        self.topk = topk
        self.base_loss = nn.MSELoss()

    def forward(self, pred, target):
        # pred/target: [B, N] 或 [B, H, W]
        loss_base = self.base_loss(pred, target)
        loss_svds = svd_topk_losses(pred, target, topk=self.topk)
        total_loss = self.base_weight * loss_base
        for w, l in zip(self.svd_weights, loss_svds):
            total_loss += w * l
        return total_loss
