import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TimeEncoder(nn.Module):
    """
    Encode discrete time step indices into a feature vector.
    Supported methods: 'embedding', 'mlp', 'concat', 'none'
    - embedding: nn.Embedding(max_time_steps, t_dim)
    - mlp: scalar t/T -> MLP -> t_dim
    - concat: scalar t/T as a single feature (t_dim = 1)
    - none: no time encoding (t_dim = 0)
    """
    def __init__(self, max_time_steps: int = 100, method: str = "embedding", t_dim: int = 32):
        super().__init__()
        method = (method or "none").lower()
        self.method = method
        self.max_time_steps = int(max_time_steps)
        if method == "embedding":
            self.embedding = nn.Embedding(self.max_time_steps, t_dim)
            self.t_dim = t_dim
        elif method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(1, t_dim),
                nn.ReLU(inplace=True),
                nn.Linear(t_dim, t_dim),
            )
            self.t_dim = t_dim
        elif method == "concat":
            self.t_dim = 1
        else:
            self.t_dim = 0

    def forward(self, time_steps: Optional[torch.Tensor]):
        if self.t_dim == 0 or time_steps is None:
            # Return empty feature to simplify downstream logic
            if time_steps is None:
                device = torch.device("cpu")
            else:
                device = time_steps.device
            return torch.zeros((time_steps.shape[0] if time_steps is not None else 1, 0), device=device)
        if self.method == "embedding":
            return self.embedding(time_steps.clamp(min=0, max=self.max_time_steps-1).long())
        elif self.method == "mlp":
            x = (time_steps.float() / max(1, self.max_time_steps-1)).unsqueeze(-1)
            return self.mlp(x)
        elif self.method == "concat":
            x = (time_steps.float() / max(1, self.max_time_steps-1)).unsqueeze(-1)
            return x
        else:
            return torch.zeros((time_steps.shape[0], 0), device=time_steps.device)


class LinearBaselineModel(nn.Module):
    """
    Simple linear regression baseline with optional time branch.
    y = W_x x + W_t phi(t)
    """
    def __init__(self, input_dim: int, output_dim: int, max_time_steps: int = 100, time_encoding: str = "none", d_model: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_enc = TimeEncoder(max_time_steps=max_time_steps, method=time_encoding, t_dim=d_model)
        self.lin_x = nn.Linear(input_dim, output_dim)
        self.lin_t = nn.Linear(self.time_enc.t_dim, output_dim) if self.time_enc.t_dim > 0 else None

    def forward(self, x_in_pressures_flat: torch.Tensor, x_time_steps: Optional[torch.Tensor]):
        y = self.lin_x(x_in_pressures_flat)
        if self.lin_t is not None:
            tfeat = self.time_enc(x_time_steps)
            y = y + self.lin_t(tfeat)
        return y


class MLPBaselineModel(nn.Module):
    """
    Two-layer MLP baseline with optional time branch added residually at hidden.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, max_time_steps: int = 100, time_encoding: str = "none"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.time_enc = TimeEncoder(max_time_steps=max_time_steps, method=time_encoding, t_dim=hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.t_to_hidden = nn.Linear(self.time_enc.t_dim, hidden_dim) if self.time_enc.t_dim > 0 else None

    def forward(self, x_in_pressures_flat: torch.Tensor, x_time_steps: Optional[torch.Tensor]):
        h = F.relu(self.fc1(x_in_pressures_flat))
        if self.t_to_hidden is not None:
            tfeat = self.time_enc(x_time_steps)
            h = h + self.t_to_hidden(tfeat)
        y = self.fc2(h)
        return y


class CNNBaselineModel(nn.Module):
    """
    Simple fully-convolutional baseline operating on 2D inputs.
    Expects model.input_hw=(H,W). Time is injected as an extra channel map with constant value t/T.
    If output_dim != input_dim and is a perfect square, the output is interpolated to sqrt(output_dim).
    """
    def __init__(self, input_dim: int, output_dim: int, input_hw: Tuple[int, int], max_time_steps: int = 100, time_encoding: str = "mlp"):
        super().__init__()
        if input_hw is None:
            raise ValueError("CNNBaselineModel requires model.input_hw (H,W).")
        self.H, self.W = int(input_hw[0]), int(input_hw[1])
        assert self.H * self.W == input_dim, f"input_dim must equal H*W, got {input_dim} vs {self.H}*{self.W}"
        # derive output spatial size
        if output_dim == input_dim:
            self.H_out, self.W_out = self.H, self.W
        else:
            s = int(math.isqrt(output_dim))
            if s * s != int(output_dim):
                raise ValueError(f"CNNBaselineModel: cannot infer square output grid from output_dim={output_dim}")
            self.H_out, self.W_out = s, s
        self.output_dim = output_dim
        self.max_time_steps = int(max_time_steps)
        self.use_time = time_encoding is not None and time_encoding.lower() != "none"
        in_ch = 1 + (1 if self.use_time else 0)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x_in_pressures_flat: torch.Tensor, x_time_steps: Optional[torch.Tensor]):
        B = x_in_pressures_flat.shape[0]
        x = x_in_pressures_flat.view(B, 1, self.H, self.W)
        if self.use_time and x_time_steps is not None:
            tnorm = (x_time_steps.float() / max(1, self.max_time_steps-1)).view(B, 1, 1, 1)
            tmap = tnorm.expand(B, 1, self.H, self.W)
            x = torch.cat([x, tmap], dim=1)
        y = self.conv(x)
        if (self.H_out, self.W_out) != (self.H, self.W):
            y = F.interpolate(y, size=(self.H_out, self.W_out), mode="bilinear", align_corners=False)
        return y.view(B, -1)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetBaselineModel(nn.Module):
    """
    Minimal 2-level U-Net with time as extra input channel.
    If output_dim != input_dim and is a perfect square, the output is interpolated to sqrt(output_dim).
    """
    def __init__(self, input_dim: int, output_dim: int, input_hw: Tuple[int, int], base_ch: int = 16, max_time_steps: int = 100, time_encoding: str = "mlp"):
        super().__init__()
        if input_hw is None:
            raise ValueError("UNetBaselineModel requires model.input_hw (H,W).")
        self.H, self.W = int(input_hw[0]), int(input_hw[1])
        assert self.H * self.W == input_dim, f"input_dim must equal H*W, got {input_dim} vs {self.H}*{self.W}"
        if output_dim == input_dim:
            self.H_out, self.W_out = self.H, self.W
        else:
            s = int(math.isqrt(output_dim))
            if s * s != int(output_dim):
                raise ValueError(f"UNetBaselineModel: cannot infer square output grid from output_dim={output_dim}")
            self.H_out, self.W_out = s, s
        self.output_dim = output_dim
        self.max_time_steps = int(max_time_steps)
        self.use_time = time_encoding is not None and time_encoding.lower() != "none"
        in_ch = 1 + (1 if self.use_time else 0)

        self.enc1 = UNetBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(base_ch, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x_in_pressures_flat: torch.Tensor, x_time_steps: Optional[torch.Tensor]):
        B = x_in_pressures_flat.shape[0]
        x = x_in_pressures_flat.view(B, 1, self.H, self.W)
        if self.use_time and x_time_steps is not None:
            tnorm = (x_time_steps.float() / max(1, self.max_time_steps-1)).view(B, 1, 1, 1)
            tmap = tnorm.expand(B, 1, self.H, self.W)
            x = torch.cat([x, tmap], dim=1)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        u1 = self.up1(e2)
        # If input size not divisible by 2, pad/crop to match
        if u1.shape[-2:] != e1.shape[-2:]:
            # center crop or pad u1 to e1
            diff_h = e1.shape[-2] - u1.shape[-2]
            diff_w = e1.shape[-1] - u1.shape[-1]
            u1 = F.pad(u1, (0, max(0, diff_w), 0, max(0, diff_h)))
            u1 = u1[:, :, :e1.shape[-2], :e1.shape[-1]]
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out = self.out_conv(d1)
        if (self.H_out, self.W_out) != (self.H, self.W):
            out = F.interpolate(out, size=(self.H_out, self.W_out), mode="bilinear", align_corners=False)
        return out.view(B, -1)


# ===== New horizontal baselines that support input_dim != output_dim =====
class CNNResizeBaselineModel(nn.Module):
    """
    Fully-convolutional baseline that can resize from input_hw to output_hw.
    - Time is injected as an extra constant channel at input resolution.
    - Uses bilinear interpolation to match the target spatial resolution.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 input_hw: Tuple[int, int], output_hw: Tuple[int, int],
                 max_time_steps: int = 100, time_encoding: str = "mlp"):
        super().__init__()
        if input_hw is None or output_hw is None:
            raise ValueError("CNNResizeBaselineModel requires model.input_hw and model.output_hw.")
        self.H_in, self.W_in = int(input_hw[0]), int(input_hw[1])
        self.H_out, self.W_out = int(output_hw[0]), int(output_hw[1])
        assert self.H_in * self.W_in == input_dim, f"input_dim must equal H_in*W_in, got {input_dim} vs {self.H_in}*{self.W_in}"
        assert self.H_out * self.W_out == output_dim, f"output_dim must equal H_out*W_out, got {output_dim} vs {self.H_out}*{self.W_out}"
        self.max_time_steps = int(max_time_steps)
        self.use_time = time_encoding is not None and time_encoding.lower() != "none"
        in_ch = 1 + (1 if self.use_time else 0)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x_in_pressures_flat: torch.Tensor, x_time_steps: Optional[torch.Tensor]):
        B = x_in_pressures_flat.shape[0]
        x = x_in_pressures_flat.view(B, 1, self.H_in, self.W_in)
        if self.use_time and x_time_steps is not None:
            tnorm = (x_time_steps.float() / max(1, self.max_time_steps-1)).view(B, 1, 1, 1)
            tmap = tnorm.expand(B, 1, self.H_in, self.W_in)
            x = torch.cat([x, tmap], dim=1)
        feat = self.backbone(x)
        feat_up = F.interpolate(feat, size=(self.H_out, self.W_out), mode="bilinear", align_corners=False)
        out = self.head(feat_up)
        return out.view(B, -1)


class UNetResizeBaselineModel(nn.Module):
    """
    U-Net style baseline that upsamples from input_hw to output_hw by a final interpolation stage.
    - Keeps the same encoder-decoder core as UNetBaselineModel but allows arbitrary output size.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 input_hw: Tuple[int, int], output_hw: Tuple[int, int],
                 base_ch: int = 16, max_time_steps: int = 100, time_encoding: str = "mlp"):
        super().__init__()
        if input_hw is None or output_hw is None:
            raise ValueError("UNetResizeBaselineModel requires model.input_hw and model.output_hw.")
        self.H_in, self.W_in = int(input_hw[0]), int(input_hw[1])
        self.H_out, self.W_out = int(output_hw[0]), int(output_hw[1])
        assert self.H_in * self.W_in == input_dim, f"input_dim must equal H_in*W_in, got {input_dim} vs {self.H_in}*{self.W_in}"
        assert self.H_out * self.W_out == output_dim, f"output_dim must equal H_out*W_out, got {output_dim} vs {self.H_out}*{self.W_out}"
        self.max_time_steps = int(max_time_steps)
        self.use_time = time_encoding is not None and time_encoding.lower() != "none"
        in_ch = 1 + (1 if self.use_time else 0)

        self.enc1 = UNetBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(base_ch, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x_in_pressures_flat: torch.Tensor, x_time_steps: Optional[torch.Tensor]):
        B = x_in_pressures_flat.shape[0]
        x = x_in_pressures_flat.view(B, 1, self.H_in, self.W_in)
        if self.use_time and x_time_steps is not None:
            tnorm = (x_time_steps.float() / max(1, self.max_time_steps-1)).view(B, 1, 1, 1)
            tmap = tnorm.expand(B, 1, self.H_in, self.W_in)
            x = torch.cat([x, tmap], dim=1)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        u1 = self.up1(e2)
        if u1.shape[-2:] != e1.shape[-2:]:
            diff_h = e1.shape[-2] - u1.shape[-2]
            diff_w = e1.shape[-1] - u1.shape[-1]
            u1 = F.pad(u1, (0, max(0, diff_w), 0, max(0, diff_h)))
            u1 = u1[:, :, :e1.shape[-2], :e1.shape[-1]]
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out_in = self.out_conv(d1)
        out = F.interpolate(out_in, size=(self.H_out, self.W_out), mode="bilinear", align_corners=False)
        return out.view(B, -1)


class PODLSEModel(nn.Module):
    """
    POD + LSE baseline model
    - Offline: compute POD basis U (output_dim x k)
    - Online (training/inference): learn a regression from input (and time) to POD coefficients c (k,)
      and reconstruct y = U @ c
    This module expects:
      forward(x_in_pressures_flat: (B, input_dim), x_time_steps: (B,)) -> y_flat: (B, output_dim)
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pod_basis: torch.Tensor,
        max_time_steps: int = 100,
        d_model: int = 512,
        time_encoding: str = "concat",
    ) -> None:
        super().__init__()
        assert pod_basis is not None and pod_basis.ndim == 2, "pod_basis must be 2D [output_dim, k]"
        self.register_buffer("U", pod_basis)  # [D_out, k]
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.k = pod_basis.shape[1]
        self.max_time_steps = max(1, int(max_time_steps))
        self.time_encoding = time_encoding

        in_features = input_dim
        if time_encoding in ("concat", "mlp"):
            in_features = input_dim + 1  # concat normalized time scalar

        # Lightweight regressor for LSE (can be linear or a tiny MLP)
        # Use a 2-layer MLP with hidden size ~ d_model//2 to be flexible
        hidden = max(32, d_model // 2)
        self.regressor = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.k),
        )

    def forward(self, x_in_pressures_flat: torch.Tensor, x_time_steps: torch.Tensor) -> torch.Tensor:
        # x_in: [B, input_dim], time: [B] or [B, 1]
        B = x_in_pressures_flat.shape[0]
        if x_time_steps.dim() == 1:
            t = x_time_steps.float().unsqueeze(-1)
        else:
            t = x_time_steps.float()
        t_norm = t / float(self.max_time_steps)

        if self.time_encoding in ("concat", "mlp"):
            x_feat = torch.cat([x_in_pressures_flat, t_norm], dim=-1)
        else:
            x_feat = x_in_pressures_flat

        coeffs = self.regressor(x_feat)  # [B, k]
        # Reconstruct in output space: y = U @ c
        # U: [D_out, k], coeffs: [B, k] -> y: [B, D_out]
        y = torch.matmul(coeffs, self.U.t())
        return y