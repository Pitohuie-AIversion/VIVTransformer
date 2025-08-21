import torch
import torch.nn as nn
import copy
import math
from .components.attention_factory import get_attention_module  # 动态获取注意力模块

# 统一导入所有可用的注意力机制
from fightingcv_attention.attention.ExternalAttention import ExternalAttention
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from fightingcv_attention.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.SKAttention import SKAttention
from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.BAM import BAMBlock
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.DANet import DAModule
from fightingcv_attention.attention.PSA import PSA
from fightingcv_attention.attention.EMSA import EMSA
from fightingcv_attention.attention.ShuffleAttention import ShuffleAttention
from fightingcv_attention.attention.MUSEAttention import MUSEAttention
from fightingcv_attention.attention.SGE import SpatialGroupEnhance
from fightingcv_attention.attention.A2Atttention import DoubleAttention
from fightingcv_attention.attention.AFT import AFT_FULL
from fightingcv_attention.attention.OutlookAttention import OutlookAttention
from fightingcv_attention.attention.ViP import WeightedPermuteMLP
from fightingcv_attention.attention.CoAtNet import CoAtNet
from fightingcv_attention.attention.HaloAttention import HaloAttention
from fightingcv_attention.attention.PolarizedSelfAttention import SequentialPolarizedSelfAttention
from fightingcv_attention.attention.CoTAttention import CoTAttention
from fightingcv_attention.attention.ResidualAttention import ResidualAttention
from fightingcv_attention.attention.S2Attention import S2Attention
from fightingcv_attention.attention.gfnet import GFNet
from fightingcv_attention.attention.TripletAttention import TripletAttention
from fightingcv_attention.attention.CoordAttention import CoordAtt
from fightingcv_attention.attention.MobileViTAttention import MobileViTAttention
from fightingcv_attention.attention.ParNetAttention import ParNetAttention
from fightingcv_attention.attention.UFOAttention import UFOAttention
# from fightingcv_attention.attention.ACmix import ACmix
from fightingcv_attention.attention.MobileViTv2Attention import MobileViTv2Attention
from fightingcv_attention.attention.DAT import DAT
from fightingcv_attention.attention.Crossformer import CrossFormer
from fightingcv_attention.attention.MOATransformer import MOATransformer
from fightingcv_attention.attention.CrissCrossAttention import CrissCrossAttention
from fightingcv_attention.attention.Axial_attention import AxialImageTransformer

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# --- 新增：自适应网格尺寸与填充的工具函数 ---
def _infer_hw_with_padding(seq_len: int, input_hw=None):
    """
    返回 (H, W), pad_len, target_len
    - 若提供 input_hw 且 H*W == seq_len，直接使用，不填充
    - 否则优先取最接近正方形的 (H, W) 使得 H*W >= seq_len（必要时进行填充）
    """
    if input_hw is not None and input_hw[0] * input_hw[1] == seq_len:
        return (int(input_hw[0]), int(input_hw[1])), 0, seq_len

    # 先尝试完美平方
    root = int(seq_len ** 0.5)
    if root * root == seq_len:
        return (root, root), 0, seq_len

    # 尝试找到最接近的因子对（若因子对极端，如 1×N，则退而求其次做“近似正方形填充”）
    factors = []
    for i in range(1, int(seq_len ** 0.5) + 1):
        if seq_len % i == 0:
            factors.append((i, seq_len // i))
    if factors:
        h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
        if min(h, w) > 1:
            return (h, w), 0, seq_len
        # 如果是 1×N 这类极端比例，改用近似正方形填充
    # 近似正方形填充：h 为 floor(sqrt(N))，w 为 ceil(N/h)
    h = max(1, int(seq_len ** 0.5))
    w = math.ceil(seq_len / h)
    target_len = h * w
    pad_len = target_len - seq_len
    return (h, w), pad_len, target_len

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, attention_type="relative", seq_len=32, input_hw=None):
        super().__init__()

        self.seq_len = seq_len
        self.input_hw = input_hw  # (H, W) 或 None

        # 选择注意力机制（将 seq_len 传给需要的模块，如 AFT）
        self.self_attn = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads, seq_len=seq_len, input_hw=self.input_hw)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化和残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        batch_size, seq_len, d_model = src.shape

        # 计算自适应 (H, W) 与必要的填充
        (spatial_dim_h, spatial_dim_w), pad_len, target_len = _infer_hw_with_padding(seq_len, self.input_hw)
        if pad_len > 0:
            pad = src.new_zeros((batch_size, pad_len, d_model))
            src_for_reshape = torch.cat([src, pad], dim=1)
        else:
            src_for_reshape = src

        # ---- Self-Attention 适配 ----
        if isinstance(self.self_attn, ExternalAttention):
            src2 = self.self_attn(src)

        elif isinstance(self.self_attn, (SEAttention, SKAttention, CBAMBlock, BAMBlock,
                                         ECAAttention, ShuffleAttention, SpatialGroupEnhance,
                                         ResidualAttention, S2Attention, TripletAttention,
                                         CoordAtt, PSA, DAModule, CoTAttention,
                                         SequentialPolarizedSelfAttention,
                                         CoAtNet,
                                         HaloAttention, DoubleAttention, ParNetAttention, )):
            # CNN类注意力机制，使用(H, W)进行重塑（必要时已做填充）
            try:
                x = src_for_reshape.transpose(1, 2).contiguous()  # [B, D, target_len]
                x = x.view(batch_size, d_model, spatial_dim_h, spatial_dim_w)
                # 准备通道自适配器（一次构建，后续复用）
                def _ensure_adapters(attn_mod, x_tensor, prefix):
                    in_adapter = getattr(self, f"{prefix}_in", None)
                    out_adapter = getattr(self, f"{prefix}_out", None)
                    # 推断期望输入/输出通道
                    in_ch = None
                    for name in ("in_channels", "in_dim", "channels", "channel", "dim"):
                        v = getattr(attn_mod, name, None)
                        if isinstance(v, int):
                            in_ch = v
                            break
                    if in_ch is None:
                        for m in attn_mod.modules():
                            if isinstance(m, nn.Conv2d):
                                in_ch = m.in_channels
                                break
                    out_ch = None
                    for name in ("out_channels", "channels", "channel", "dim"):
                        v = getattr(attn_mod, name, None)
                        if isinstance(v, int):
                            out_ch = v
                            break
                    if out_ch is None:
                        for m in reversed(list(attn_mod.modules())):
                            if isinstance(m, nn.Conv2d):
                                out_ch = m.out_channels
                                break
                    if out_ch is None:
                        out_ch = in_ch if in_ch is not None else d_model
                    # 创建并注册adapter
                    if in_ch is not None and in_ch != d_model and in_adapter is None:
                        in_adapter = nn.Conv2d(d_model, in_ch, kernel_size=1, bias=False).to(x_tensor.device)
                        setattr(self, f"{prefix}_in", in_adapter)
                    if out_ch is not None and out_ch != d_model and out_adapter is None:
                        out_adapter = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False).to(x_tensor.device)
                        setattr(self, f"{prefix}_out", out_adapter)
                    return getattr(self, f"{prefix}_in", None), getattr(self, f"{prefix}_out", None)
                in_adp, out_adp = _ensure_adapters(self.self_attn, x, "_enc_sa2d_adapter")
                # 对齐设备后再调用模块
                orig_device = x.device
                try:
                    param = next(self.self_attn.parameters())
                    module_device = param.device
                except StopIteration:
                    module_device = orig_device
                if module_device != orig_device:
                    x = x.to(module_device)
                if in_adp is not None and in_adp.weight.device != x.device:
                    in_adp = in_adp.to(x.device)
                if out_adp is not None and out_adp.weight.device != x.device:
                    out_adp = out_adp.to(x.device)
                if in_adp is not None:
                    x = in_adp(x)
                x = self.self_attn(x)
                if out_adp is not None:
                    x = out_adp(x)
                if x.device != orig_device:
                    x = x.to(orig_device)
                x = x.view(batch_size, d_model, -1)  # [B, D, target_len]
                src2 = x[:, :, :seq_len].transpose(1, 2)
            except Exception as e:
                print(f"Warning: Encoder attention reshape/adapt failed ({type(e).__name__}: {e}), using identity fallback")
                src2 = src

        elif isinstance(self.self_attn, (AFT_FULL,)):
            src2 = self.self_attn(src)

        elif isinstance(self.self_attn, (OutlookAttention, WeightedPermuteMLP)):
            # 这些模块期望输入为 [B, H, W, C]。ViP 额外要求 C 能被 H(=W) 整除，且构造时 seg_dim==H
            x = src_for_reshape.contiguous().view(batch_size, spatial_dim_h, spatial_dim_w, d_model)
            try:
                x = self.self_attn(x)
            except RuntimeError as e:
                # 若 attention_factory 回退为 Identity，则直接透传；否则打印一次告警
                if not isinstance(self.self_attn, nn.Identity):
                    print(f"Warning: ViP/Outlook forward failed ({e}); falling back to identity for this pass")
                x = src_for_reshape.view(batch_size, spatial_dim_h, spatial_dim_w, d_model)
            x = x.view(batch_size, -1, d_model)  # [B, target_len, D]
            src2 = x[:, :seq_len, :]

        elif isinstance(self.self_attn, nn.Identity):
            # 恒等映射：直接透传
            src2 = src

        else:
            # 标准 Transformer 接口 (Q,K,V)
            src2 = self.self_attn(src, src, src)

        # 残差连接 + LayerNorm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward 网络
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, attention_type="relative", seq_len=32, input_hw=None):
        super().__init__()

        self.seq_len = seq_len
        self.input_hw = input_hw

        # 选择注意力机制
        self.self_attn = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads, seq_len=seq_len, input_hw=self.input_hw)
        self.multihead_attn = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads, seq_len=seq_len, input_hw=self.input_hw)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化和残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        batch_size, seq_len, d_model = tgt.shape

        # 计算自适应 (H, W) 与必要的填充
        (spatial_dim_h, spatial_dim_w), pad_len, target_len = _infer_hw_with_padding(seq_len, self.input_hw)
        if pad_len > 0:
            pad_t = tgt.new_zeros((batch_size, pad_len, d_model))
            pad_m = memory.new_zeros((batch_size, pad_len, d_model))
            tgt_for_reshape = torch.cat([tgt, pad_t], dim=1)
            memory_for_reshape = torch.cat([memory, pad_m], dim=1)
        else:
            tgt_for_reshape = tgt
            memory_for_reshape = memory

        # ---- Self-Attention 适配 ----
        if isinstance(self.self_attn, ExternalAttention):
            tgt2 = self.self_attn(tgt)

        elif isinstance(self.self_attn, (SEAttention, SKAttention, CBAMBlock, BAMBlock,
                                         ECAAttention, ShuffleAttention, SpatialGroupEnhance,
                                         ResidualAttention, S2Attention, TripletAttention,
                                         CoordAtt, PSA, DAModule, CoTAttention,
                                         SequentialPolarizedSelfAttention,
                                         CoAtNet,
                                         HaloAttention, DoubleAttention, ParNetAttention, )):
            try:
                x = tgt_for_reshape.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim_h, spatial_dim_w)
                # 通道自适配器（Decoder Self）
                def _ensure_adapters(attn_mod, x_tensor, prefix):
                    in_adapter = getattr(self, f"{prefix}_in", None)
                    out_adapter = getattr(self, f"{prefix}_out", None)
                    in_ch = None
                    for name in ("in_channels", "in_dim", "channels", "channel", "dim"):
                        v = getattr(attn_mod, name, None)
                        if isinstance(v, int):
                            in_ch = v
                            break
                    if in_ch is None:
                        for m in attn_mod.modules():
                            if isinstance(m, nn.Conv2d):
                                in_ch = m.in_channels
                                break
                    out_ch = None
                    for name in ("out_channels", "channels", "channel", "dim"):
                        v = getattr(attn_mod, name, None)
                        if isinstance(v, int):
                            out_ch = v
                            break
                    if out_ch is None:
                        for m in reversed(list(attn_mod.modules())):
                            if isinstance(m, nn.Conv2d):
                                out_ch = m.out_channels
                                break
                    if out_ch is None:
                        out_ch = in_ch if in_ch is not None else d_model
                    if in_ch is not None and in_ch != d_model and in_adapter is None:
                        in_adapter = nn.Conv2d(d_model, in_ch, kernel_size=1, bias=False).to(x_tensor.device)
                        setattr(self, f"{prefix}_in", in_adapter)
                    if out_ch is not None and out_ch != d_model and out_adapter is None:
                        out_adapter = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False).to(x_tensor.device)
                        setattr(self, f"{prefix}_out", out_adapter)
                    return getattr(self, f"{prefix}_in", None), getattr(self, f"{prefix}_out", None)
                in_adp, out_adp = _ensure_adapters(self.self_attn, x, "_dec_self_sa2d_adapter")
                orig_device = x.device
                try:
                    param = next(self.self_attn.parameters())
                    module_device = param.device
                except StopIteration:
                    module_device = orig_device
                if module_device != orig_device:
                    x = x.to(module_device)
                if in_adp is not None and in_adp.weight.device != x.device:
                    in_adp = in_adp.to(x.device)
                if out_adp is not None and out_adp.weight.device != x.device:
                    out_adp = out_adp.to(x.device)
                if in_adp is not None:
                    x = in_adp(x)
                x = self.self_attn(x)
                if out_adp is not None:
                    x = out_adp(x)
                if x.device != orig_device:
                    x = x.to(orig_device)
                x = x.view(batch_size, d_model, -1)
                tgt2 = x[:, :, :seq_len].transpose(1, 2)
            except Exception as e:
                print(f"Warning: ViP/CNN-like decoder self attention failed ({type(e).__name__}: {e}); identity fallback")
                tgt2 = tgt

        elif isinstance(self.self_attn, (AFT_FULL,)):
            tgt2 = self.self_attn(tgt)

        elif isinstance(self.self_attn, (OutlookAttention, WeightedPermuteMLP)):
            # ViP/Outlook: 输入 [B, H, W, C]，ViP 要求 seg_dim==H==W
            x = tgt_for_reshape.view(batch_size, spatial_dim_h, spatial_dim_w, d_model)
            try:
                x = self.self_attn(x)
            except RuntimeError as e:
                if not isinstance(self.self_attn, nn.Identity):
                    print(f"Warning: ViP/Outlook forward (decoder self) failed ({e}); identity fallback")
                x = tgt_for_reshape.view(batch_size, spatial_dim_h, spatial_dim_w, d_model)
            x = x.view(batch_size, -1, d_model)
            tgt2 = x[:, :seq_len, :]

        elif isinstance(self.self_attn, nn.Identity):
            # 恒等映射：直接透传
            tgt2 = tgt

        else:
            tgt2 = self.self_attn(tgt, tgt, tgt)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 交叉注意力（multihead_attn）
        if isinstance(self.multihead_attn, (SEAttention, SKAttention, CBAMBlock, BAMBlock,
                                            ECAAttention, ShuffleAttention, SpatialGroupEnhance,
                                            ResidualAttention, S2Attention, TripletAttention,
                                            CoordAtt, PSA, DAModule, CoTAttention,
                                            SequentialPolarizedSelfAttention,
                                            CoAtNet,
                                            HaloAttention, DoubleAttention, ParNetAttention)):
            try:
                x = tgt_for_reshape.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim_h, spatial_dim_w)
                # 通道自适配器（Decoder Cross）
                def _ensure_adapters(attn_mod, x_tensor, prefix):
                    in_adapter = getattr(self, f"{prefix}_in", None)
                    out_adapter = getattr(self, f"{prefix}_out", None)
                    in_ch = None
                    for name in ("in_channels", "in_dim", "channels", "channel", "dim"):
                        v = getattr(attn_mod, name, None)
                        if isinstance(v, int):
                            in_ch = v
                            break
                    if in_ch is None:
                        for m in attn_mod.modules():
                            if isinstance(m, nn.Conv2d):
                                in_ch = m.in_channels
                                break
                    out_ch = None
                    for name in ("out_channels", "channels", "channel", "dim"):
                        v = getattr(attn_mod, name, None)
                        if isinstance(v, int):
                            out_ch = v
                            break
                    if out_ch is None:
                        for m in reversed(list(attn_mod.modules())):
                            if isinstance(m, nn.Conv2d):
                                out_ch = m.out_channels
                                break
                    if out_ch is None:
                        out_ch = in_ch if in_ch is not None else d_model
                    if in_ch is not None and in_ch != d_model and in_adapter is None:
                        in_adapter = nn.Conv2d(d_model, in_ch, kernel_size=1, bias=False).to(x_tensor.device)
                        setattr(self, f"{prefix}_in", in_adapter)
                    if out_ch is not None and out_ch != d_model and out_adapter is None:
                        out_adapter = nn.Conv2d(out_ch, d_model, kernel_size=1, bias=False).to(x_tensor.device)
                        setattr(self, f"{prefix}_out", out_adapter)
                    return getattr(self, f"{prefix}_in", None), getattr(self, f"{prefix}_out", None)
                in_adp, out_adp = _ensure_adapters(self.multihead_attn, x, "_dec_cross_sa2d_adapter")
                orig_device = x.device
                try:
                    param = next(self.multihead_attn.parameters())
                    module_device = param.device
                except StopIteration:
                    module_device = orig_device
                if module_device != orig_device:
                    x = x.to(module_device)
                if in_adp is not None and in_adp.weight.device != x.device:
                    in_adp = in_adp.to(x.device)
                if out_adp is not None and out_adp.weight.device != x.device:
                    out_adp = out_adp.to(x.device)
                if in_adp is not None:
                    x = in_adp(x)
                x = self.multihead_attn(x)
                if out_adp is not None:
                    x = out_adp(x)
                if x.device != orig_device:
                    x = x.to(orig_device)
                x = x.view(batch_size, d_model, -1)
                tgt2 = x[:, :, :seq_len].transpose(1, 2)
            except Exception as e:
                print(f"Warning: ViP/CNN-like decoder cross attention failed ({type(e).__name__}: {e}); identity fallback")
                tgt2 = tgt
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class CustomEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
class CustomDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt

class TransformerFlowReconstructionModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=6,
                 d_model=512, max_time_steps=100, attention_type="relative", seq_len=32, input_hw=None):
        super().__init__()

        self.attention_type = attention_type
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 若未显式给出 input_hw，则根据 input_dim 自动推断 (尽量接近正方形且 H*W == input_dim)
        if input_hw is None:
            # 优先尝试完美平方
            root = int(input_dim ** 0.5)
            if root * root == input_dim:
                input_hw = (root, root)
            else:
                # 寻找最接近的因子对
                factors = []
                for i in range(1, int(input_dim ** 0.5) + 1):
                    if input_dim % i == 0:
                        factors.append((i, input_dim // i))
                if factors:
                    input_hw = min(factors, key=lambda x: abs(x[0] - x[1]))
                else:
                    input_hw = (1, input_dim)
        # 记录 input_hw，并用其覆盖 seq_len
        self.input_hw = input_hw  # (H, W)
        self.seq_len = input_hw[0] * input_hw[1]

        # 确保embedding输出维度正确：将 input_dim 投影为 (seq_len * d_model)
        self.embedding = nn.Linear(input_dim, self.seq_len * d_model)
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))

        encoder_layer = CustomEncoderLayer(d_model=d_model, num_heads=num_heads,
                                           dim_feedforward=2048, attention_type=attention_type,
                                           seq_len=self.seq_len, input_hw=self.input_hw)
        decoder_layer = CustomDecoderLayer(d_model=d_model, num_heads=num_heads,
                                           dim_feedforward=2048, attention_type=attention_type,
                                           seq_len=self.seq_len, input_hw=self.input_hw)

        self.encoder = CustomEncoder(encoder_layer, num_layers)
        self.decoder = CustomDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x_in_pressures_flat, x_time_steps):
        batch_size = x_in_pressures_flat.size(0)
        x_time_steps = x_time_steps.long()

        # 确保时间步索引在有效范围内（根据embedding大小动态确定）
        max_idx = self.time_step_embedding.num_embeddings - 1
        x_time_steps = torch.clamp(x_time_steps, 0, max_idx)

        # 使用模型推断/指定的 seq_len
        x_embedded = self.embedding(x_in_pressures_flat)  # [batch_size, seq_len * d_model]

        # 计算正确的d_model维度
        expected_d_model = x_embedded.size(1) // self.seq_len
        
        # reshape 为 [batch_size, seq_len, d_model]
        x_embedded = x_embedded.view(batch_size, self.seq_len, expected_d_model)

        # 加上时间步和位置编码
        time_emb = self.time_step_embedding(x_time_steps).unsqueeze(1)  # [batch_size, 1, d_model]
        pos_enc = self.positional_encoding.expand(batch_size, 1, -1)  # [batch_size, 1, d_model]
        
        # 确保维度匹配
        if time_emb.size(-1) != expected_d_model:
            time_emb = time_emb[:, :, :expected_d_model]
        if pos_enc.size(-1) != expected_d_model:
            pos_enc = pos_enc[:, :, :expected_d_model]
            
        x_embedded = x_embedded + time_emb + pos_enc

        encoder_output = self.encoder(x_embedded)
        decoder_output = self.decoder(encoder_output, encoder_output)

        out_pressure_flat_pred = self.fc_out(decoder_output.mean(dim=1))

        return out_pressure_flat_pred


