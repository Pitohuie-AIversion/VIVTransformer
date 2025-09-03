import torch
import torch.nn as nn
import copy
import math
import logging
from typing import Optional
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

# Module-level logger
_logger = logging.getLogger(__name__)

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
                _logger.warning("Encoder attention reshape/adapt failed (%s: %s), using identity fallback", type(e).__name__, e)
                src2 = src

        elif isinstance(self.self_attn, (AFT_FULL,)):
            src2 = self.self_attn(src)

        elif isinstance(self.self_attn, (OutlookAttention, WeightedPermuteMLP)):
            # 这些模块期望输入为 [B, H, W, C]。ViP 要求 seg_dim==H==W
            x = src_for_reshape.contiguous().view(batch_size, spatial_dim_h, spatial_dim_w, d_model)
            try:
                x = self.self_attn(x)
            except RuntimeError as e:
                # 若 attention_factory 回退为 Identity，则直接透传；否则打印一次告警
                if not isinstance(self.self_attn, nn.Identity):
                    _logger.warning("ViP/Outlook forward failed (%s); falling back to identity for this pass", e)
                x = src_for_reshape.view(batch_size, spatial_dim_h, spatial_dim_w, d_model)
            x = x.view(batch_size, -1, d_model)  # [B, target_len, D]
            src2 = x[:, :seq_len, :]

        elif isinstance(self.self_attn, nn.Identity):
            # 恒等映射：直接透传
            src2 = self.norm1(src)  # 与 Pre-LN 对齐，虽然恒等，但保持范式

        else:
            # 标准 Transformer 接口 (Q,K,V)
            norm_src = self.norm1(src)
            src2 = self.self_attn(norm_src, norm_src, norm_src)

        # 残差连接（Pre-LN：Norm 在前）
        src = src + self.dropout1(src2)

        # Feed-forward 网络（Pre-LN）
        norm_src = self.norm2(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(norm_src))))
        src = src + self.dropout2(src2)

        return src


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, attention_type="relative", seq_len=32, input_hw=None, use_memory_film: bool = True, use_memory_concat: bool = False):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_hw = input_hw

        # 选择注意力机制
        self.self_attn = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads, seq_len=seq_len, input_hw=self.input_hw)
        self.multihead_attn = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads, seq_len=seq_len, input_hw=self.input_hw)
        # 强制标准交叉注意力（稳定可靠的回退分支）
        self.cross_attn_std = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # P0-2: 为 ViP/CNN 类路径增加 memory 融合（FiLM 门控，可配置）
        self._use_memory_film = use_memory_film
        reduce = max(1, d_model // 4)
        self._film_mlp = nn.Sequential(
            nn.Linear(d_model, reduce),
            nn.ReLU(inplace=True),
            nn.Linear(reduce, 2 * d_model),
        )
        # 新增：记忆融合 concat + 1x1 conv（线性层），适用于通用 [B,L,D] 场景
        self._use_memory_concat = use_memory_concat
        if self._use_memory_concat:
            self._concat_proj = nn.Linear(2 * d_model, d_model)
        # 标注：交叉注意力不走 2D 重塑路径（Route B）。仅告警一次。
        self._printed_cross2d_na = False
        # 统计信息
        self._stats = {
            'decoder_self_identity_fallback': 0,
            'decoder_cross_attn_notes': 0,
            'decoder_cross_attn_std_calls': 0,
        }

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

        # 计算自适应 (H, W) 与必要的填充（仅在需要做 2D 形变的路径时使用）
        (spatial_dim_h, spatial_dim_w), pad_len, target_len = _infer_hw_with_padding(seq_len, self.input_hw)
        need_2d_self = isinstance(self.self_attn, (SEAttention, SKAttention, CBAMBlock, BAMBlock,
                                                   ECAAttention, ShuffleAttention, SpatialGroupEnhance,
                                                   ResidualAttention, S2Attention, TripletAttention,
                                                   CoordAtt, PSA, DAModule, CoTAttention,
                                                   SequentialPolarizedSelfAttention,
                                                   CoAtNet,
                                                   HaloAttention, DoubleAttention, ParNetAttention,
                                                   OutlookAttention, WeightedPermuteMLP))
        if need_2d_self and pad_len > 0:
            pad_t = tgt.new_zeros((batch_size, pad_len, d_model))
            tgt_for_reshape = torch.cat([tgt, pad_t], dim=1)
        else:
            tgt_for_reshape = tgt

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
                    if in_adapter is None:
                        in_adapter = nn.Conv2d(d_model, getattr(attn_mod, "in_channels", d_model), kernel_size=1)
                        setattr(self, f"{prefix}_in", in_adapter)
                    if out_adapter is None:
                        out_adapter = nn.Conv2d(getattr(attn_mod, "out_channels", getattr(attn_mod, "channels", d_model)), d_model, kernel_size=1)
                        setattr(self, f"{prefix}_out", out_adapter)
                    return in_adapter, out_adapter
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
                self._stats['decoder_self_identity_fallback'] += 1
                _logger.warning("ViP/CNN-like decoder self attention failed (%s: %s); identity fallback", type(e).__name__, e)
                tgt2 = self.norm1(tgt)

        elif isinstance(self.self_attn, (AFT_FULL,)):
            tgt2 = self.self_attn(tgt)

        elif isinstance(self.self_attn, (OutlookAttention, WeightedPermuteMLP)):
            # ViP/Outlook: 输入 [B, H, W, C]，ViP 要求 seg_dim==H==W
            x = tgt_for_reshape.view(batch_size, spatial_dim_h, spatial_dim_w, d_model)
            try:
                x = self.self_attn(x)
            except RuntimeError as e:
                if not isinstance(self.self_attn, nn.Identity):
                    self._stats['decoder_self_identity_fallback'] += 1
                    _logger.warning("ViP/Outlook forward (decoder self) failed (%s); identity fallback", e)
                x = tgt_for_reshape.view(batch_size, spatial_dim_h, spatial_dim_w, d_model)
            x = x.view(batch_size, -1, d_model)
            tgt2 = x[:, :seq_len, :]

        elif isinstance(self.self_attn, nn.Identity):
            # 恒等映射：直接透传
            tgt2 = self.norm1(tgt)

        else:
            norm_tgt = self.norm1(tgt)
            tgt2 = self.self_attn(norm_tgt, norm_tgt, norm_tgt)

        tgt = tgt + self.dropout1(tgt2)

        # P0-2: 对 CNN/ViP 类 attention 类型，使用 memory 的全局统计进行 FiLM 调制，作为交叉注意力的 Query
        is_vip_like = isinstance(self.multihead_attn, (SEAttention, SKAttention, CBAMBlock, BAMBlock,
                                                       ECAAttention, ShuffleAttention, SpatialGroupEnhance,
                                                       ResidualAttention, S2Attention, TripletAttention,
                                                       CoordAtt, PSA, DAModule, CoTAttention,
                                                       SequentialPolarizedSelfAttention,
                                                       CoAtNet,
                                                       HaloAttention, DoubleAttention, ParNetAttention,
                                                       OutlookAttention, WeightedPermuteMLP))
        
        # 基于记忆的融合：先可选 concat+1x1 conv，再可选 FiLM（针对 ViP/CNN 类）
        tgt_mod = tgt
        if self._use_memory_concat and memory is not None and memory.ndim == 3 and memory.shape == tgt.shape:
            # 逐 token 对齐拼接后用 1x1 conv (Linear) 压回 d_model
            tgt_mod = torch.cat([tgt_mod, memory], dim=-1)
            tgt_mod = self._concat_proj(tgt_mod)
        
        if self._use_memory_film and is_vip_like and memory is not None and memory.ndim == 3 and memory.shape[-1] == d_model:
            mem_global = memory.mean(dim=1)  # [B, D]
            gamma_beta = self._film_mlp(mem_global)  # [B, 2D]
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            gamma = torch.sigmoid(gamma)
            tgt_for_cross = tgt_mod * gamma.unsqueeze(1) + beta.unsqueeze(1)
        else:
            tgt_for_cross = tgt_mod

        # Route B 说明：交叉注意力不做 2D 重塑/填充，仅接受 [B, L, D] 形状的 memory（K=V）
        if is_vip_like and not self._printed_cross2d_na:
            _logger.info("Note: Cross-attention uses standard MHA with [B,L,D] memory; 2D reshape/pad path is not applied in decoder cross-attn (Route B).")
            self._printed_cross2d_na = True
            self._stats['decoder_cross_attn_notes'] += 1
        assert memory is not None and memory.ndim == 3 and memory.shape[-1] == d_model, \
            "Decoder cross-attention expects memory in shape [B, L, D]; 2D reshape path is not supported here (Route B)."

        # 交叉注意力（强制标准 MHA：Q=tgt_for_cross, K=V=memory），Pre-LN
        norm_q = self.norm2(tgt_for_cross)
        norm_kv = memory  # memory 由 Encoder 产出，已是 [B,L,D]
        tgt2, _ = self.cross_attn_std(norm_q, norm_kv, norm_kv)
        self._stats['decoder_cross_attn_std_calls'] += 1
        tgt = tgt + self.dropout2(tgt2)

        # Feed-forward（Pre-LN）
        norm_tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(norm_tgt))))
        tgt = tgt + self.dropout3(tgt2)

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
                 d_model=512, max_time_steps=100, attention_type="relative", seq_len=32, input_hw=None, pe_type: str = "learnable_1d",
                 output_head_type: str = "global", out_channels_per_token: Optional[int] = None,
                 time_encoding: str = "embedding", use_memory_film: bool = True, use_memory_concat: bool = False):
        super().__init__()
        
        self.attention_type = attention_type
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_head_type = output_head_type  # 'global' or 'per_token'
        self._declared_per_token_channels = out_channels_per_token

        # 根据优先级确定序列长度与网格：input_hw > seq_len > 从 input_dim 推断
        self.input_hw = input_hw  # (H, W) 或 None
        if self.input_hw is not None:
            self.seq_len = self.input_hw[0] * self.input_hw[1]
        elif seq_len is not None:
            self.seq_len = seq_len
            # 如果使用 2D 可学习位置编码而未给出 input_hw，则从 seq_len 分解出一个近似方形网格
            if pe_type == 'learnable_2d' and self.input_hw is None:
                root = int(seq_len ** 0.5)
                if root * root == seq_len:
                    self.input_hw = (root, root)
                else:
                    factors = []
                    for i in range(1, int(seq_len ** 0.5) + 1):
                        if seq_len % i == 0:
                            factors.append((i, seq_len // i))
                    self.input_hw = min(factors, key=lambda x: abs(x[0] - x[1])) if factors else (seq_len, 1)
        else:
            # 若未显式给出，回退为基于 input_dim 的近似正方形网格
            root = int(input_dim ** 0.5)
            if root * root == input_dim:
                self.input_hw = (root, root)
            else:
                factors = []
                for i in range(1, int(input_dim ** 0.5) + 1):
                    if input_dim % i == 0:
                        factors.append((i, input_dim // i))
                if factors:
                    self.input_hw = min(factors, key=lambda x: abs(x[0] - x[1]))
                else:
                    self.input_hw = (1, input_dim)
            self.seq_len = self.input_hw[0] * self.input_hw[1]

        # 投影输入到 [B, L, D]
        self.embedding = nn.Linear(input_dim, self.seq_len * d_model)
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        # 时间编码方式：'embedding'（当前默认）或 'mlp'（与 origin 连续时间一致）
        self.time_encoding = time_encoding
        if self.time_encoding == 'mlp':
            self.time_mlp = nn.Sequential(
                nn.Linear(1, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )

        # P1-1 位置编码：支持 1D 可学习/正弦 与 2D 可分离可学习
        self.pe_type = pe_type  # 'learnable_1d' | 'sinusoidal_1d' | 'learnable_2d'
        # 防呆断言：当选择 2D 可分离位置编码时，必须有有效的网格 input_hw 且与 seq_len 一致
        if self.pe_type == 'learnable_2d':
            assert self.input_hw is not None, "learnable_2d PE requires input_hw to be set or inferrable"
            H, W = self.input_hw
            assert H > 0 and W > 0, "input_hw must have positive H and W for learnable_2d PE"
            assert H * W == self.seq_len, f"input_hw product H*W ({H*W}) must equal seq_len ({self.seq_len}) for learnable_2d PE"
        if self.pe_type == 'learnable_1d':
            self.positional_encoding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        elif self.pe_type == 'sinusoidal_1d':
            L, D = self.seq_len, d_model
            pe = torch.zeros(1, L, D)
            position = torch.arange(0, L, dtype=torch.float32).unsqueeze(1)  # [L,1]
            div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / D)))
            pe[:, :, 0::2] = torch.sin(position * div_term)
            if D > 1:
                pe[:, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('positional_encoding', pe, persistent=False)
        elif self.pe_type == 'learnable_2d':
            D_h = d_model // 2
            D_w = d_model - D_h
            # 若仍未有网格形状，按 (L,1) 退化为 1D 栈（保证维度合法）
            if self.input_hw is None:
                self.input_hw = (self.seq_len, 1)
            H, W = self.input_hw
            self.pe_h = nn.Parameter(torch.zeros(1, H, 1, D_h))  # [1,H,1,Dh]
            self.pe_w = nn.Parameter(torch.zeros(1, 1, W, D_w))  # [1,1,W,Dw]
        else:
            # 回退到 1D 可学习
            self.positional_encoding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        # 构建 Encoder / Decoder 堆叠
        encoder_layer = CustomEncoderLayer(d_model=d_model, num_heads=num_heads,
                                           dim_feedforward=2048, attention_type=attention_type,
                                           seq_len=self.seq_len, input_hw=self.input_hw)
        decoder_layer = CustomDecoderLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=2048,
                                           dropout=0.1, attention_type=attention_type,
                                           seq_len=self.seq_len, input_hw=self.input_hw, use_memory_film=use_memory_film,
                                           use_memory_concat=use_memory_concat)
        self.encoder = CustomEncoder(encoder_layer, num_layers)
        self.decoder = CustomDecoder(decoder_layer, num_layers)

        # 输出头：全局/逐 token
        if self.output_head_type == 'per_token':
            per_token_channels = self._declared_per_token_channels
            if per_token_channels is None:
                if output_dim % self.seq_len == 0:
                    per_token_channels = output_dim // self.seq_len
                else:
                    per_token_channels = None
            # 仅当 C 有效且 L*C 与 output_dim 一致时才创建逐 token 头，否则回退为 global
            if per_token_channels is not None and per_token_channels > 0 and (self.seq_len * per_token_channels == output_dim):
                self.token_head = nn.Linear(d_model, per_token_channels)
                self.per_token_channels = per_token_channels
            else:
                _logger.warning("per_token 配置无效（未提供 C 或 L*C != output_dim），回退为 global 头")
                self.output_head_type = 'global'
                self.fc_out = nn.Linear(d_model, output_dim)
        else:
            self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x_in_pressures_flat, x_time_steps):
        batch_size = x_in_pressures_flat.size(0)
        x_time_steps = x_time_steps.long()

        # 时间步索引裁剪至有效范围
        max_idx = self.time_step_embedding.num_embeddings - 1
        x_time_steps = torch.clamp(x_time_steps, 0, max_idx)

        # 投影并重塑为 [B, L, D]
        x_embedded = self.embedding(x_in_pressures_flat)  # [B, L*D]
        expected_d_model = x_embedded.size(1) // self.seq_len
        x_embedded = x_embedded.view(batch_size, self.seq_len, expected_d_model)

        # 时间步与位置编码
        if self.time_encoding == 'embedding':
            # 离散时间索引
            if x_time_steps.dim() == 2 and x_time_steps.size(-1) == 1:
                x_time_steps = x_time_steps.squeeze(-1)
            x_time_steps = x_time_steps.long()
            max_idx = self.time_step_embedding.num_embeddings - 1
            x_time_steps = torch.clamp(x_time_steps, 0, max_idx)
            time_emb = self.time_step_embedding(x_time_steps).unsqueeze(1)  # [B,1,D]
        else:
            # 连续时间 MLP
            if x_time_steps.dim() == 1:
                x_time_steps = x_time_steps.unsqueeze(-1)
            time_emb = self.time_mlp(x_time_steps.float()).unsqueeze(1)  # [B,1,D]
        
        if self.pe_type in ('learnable_1d', 'sinusoidal_1d'):
            pos_enc = self.positional_encoding.expand(batch_size, -1, -1)  # [B,L,D]
        elif self.pe_type == 'learnable_2d':
            H, W = self.input_hw
            D_h = self.pe_h.size(-1)
            D_w = self.pe_w.size(-1)
            pe2d = torch.cat([
                self.pe_h.expand(1, H, W, D_h),  # [1,H,W,Dh]
                self.pe_w.expand(1, 1, W, D_w).expand(1, H, W, D_w),  # [1,1,W,Dw]
            ], dim=-1).view(1, H * W, D_h + D_w)  # [1,L,D]
            pos_enc = pe2d.expand(batch_size, -1, -1)
        else:
            pos_enc = torch.zeros(batch_size, self.seq_len, expected_d_model, device=x_embedded.device)

        if time_emb.size(-1) != expected_d_model:
            time_emb = time_emb[:, :, :expected_d_model]
        if pos_enc.size(-1) != expected_d_model:
            pos_enc = pos_enc[:, :, :expected_d_model]

        x_embedded = x_embedded + time_emb + pos_enc

        # 编解码
        encoder_output = self.encoder(x_embedded)
        decoder_output = self.decoder(encoder_output, encoder_output)  # [B, L, D]

        # 输出头
        if self.output_head_type == 'per_token' and hasattr(self, 'token_head'):
            token_pred = self.token_head(decoder_output)  # [B, L, C]
            out = token_pred.reshape(batch_size, -1)      # [B, L*C]
            if out.size(1) == self.output_dim:
                return out
            else:
                # 兜底：若发生不一致，确保存在全局头回退
                if not hasattr(self, 'fc_out'):
                    self.fc_out = nn.Linear(decoder_output.size(-1), self.output_dim)
                _logger.warning("逐 token 输出维度与 output_dim 不匹配，回退为 global 头")
                return self.fc_out(decoder_output.mean(dim=1))
        else:
            out_pressure_flat_pred = self.fc_out(decoder_output.mean(dim=1))
            return out_pressure_flat_pred



    def per_token_output_to_2d(self, y_flat):
        """
        Reshape per-token flat output [B, L*C] to [B, H, W, C].
        Preconditions:
        - output_head_type == 'per_token' and token_head exists
        - input_hw is not None and seq_len == H*W
        - y_flat shape is [B, L*C] and matches configured L and C
        """
        assert self.output_head_type == 'per_token' and hasattr(self, 'token_head') and hasattr(self, 'per_token_channels'), \
            "per_token_output_to_2d requires model with per_token head configured"
        assert self.input_hw is not None and isinstance(self.input_hw, (tuple, list)) and len(self.input_hw) == 2, \
            "input_hw must be set as (H, W) to reshape per-token outputs"
        H, W = int(self.input_hw[0]), int(self.input_hw[1])
        assert H > 0 and W > 0, "input_hw must contain positive integers"
        assert self.seq_len == H * W, "seq_len must equal H*W for 2D reshape"
        assert y_flat.dim() == 2, "y_flat must be 2D tensor [B, L*C]"
        B = y_flat.size(0)
        L = self.seq_len
        C = int(self.per_token_channels)
        assert y_flat.size(1) == L * C, f"y_flat last dim must be L*C={L*C}, got {y_flat.size(1)}"
        return y_flat.view(B, L, C).view(B, H, W, C)


