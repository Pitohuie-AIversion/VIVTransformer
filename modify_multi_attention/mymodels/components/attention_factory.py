import torch.nn as nn
import math
from .attention import RelativePositionSelfAttention, SparseSelfAttention, LSHSelfAttention

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

# 创建一个字典来映射 attention_type 到实际的注意力类
ATTENTION_MODULES = {
    "external": ExternalAttention,
    "self": ScaledDotProductAttention,
    "simplified_self": SimplifiedScaledDotProductAttention,
    "se": SEAttention,
    "sk": SKAttention,
    "cbam": CBAMBlock,
    "bam": BAMBlock,
    "eca": ECAAttention,
    "danet": DAModule,
    "psa": PSA,
    "emsa": EMSA,
    "shuffle": ShuffleAttention,
    "muse": MUSEAttention,
    "sge": SpatialGroupEnhance,
    "a2": DoubleAttention,
    "aft": AFT_FULL,
    "outlook": OutlookAttention,
    "vip": WeightedPermuteMLP,
    "coatnet": CoAtNet,
    "halo": HaloAttention,
    "polarized": SequentialPolarizedSelfAttention,
    "cot": CoTAttention,
    "residual": ResidualAttention,
    "s2": S2Attention,
    "gfnet": GFNet,
    "triplet": TripletAttention,
    "coord": CoordAtt,
    "mobilevit": MobileViTAttention,
    "parnet": ParNetAttention,
    "ufo": UFOAttention,
    # "acmix": ACmix,
    "mobilevitv2": MobileViTv2Attention,
    "dat": DAT,
    "crossformer": CrossFormer,
    "moa": MOATransformer,
    "crisscross": CrissCrossAttention,
    "axial": AxialImageTransformer,
    "relative": RelativePositionSelfAttention,
    "sparse": SparseSelfAttention,
    "lsh": LSHSelfAttention,
}

# 解析 (H,W)

def _resolve_hw(seq_len=None, input_hw=None):
    if input_hw is not None:
        return int(input_hw[0]), int(input_hw[1])
    if seq_len is None:
        return 7, 7
    root = int(seq_len ** 0.5)
    if root * root == seq_len:
        return root, root
    h = max(1, root)
    w = math.ceil(seq_len / h)
    return h, w


# 选择与通道数匹配的安全分组，避免 groups/G 为 0 或不整除导致崩溃

def _safe_groups(channels: int, preferred: int = 8) -> int:
    for g in (preferred, 4, 2, 1):
        if channels >= g and channels % g == 0:
            return g
    return 1


def get_attention_module(attention_type, d_model=512, num_heads=8, **kwargs):
    """
    根据 attention_type 返回对应的注意力模块，适配不同注意力模块的参数
    """
    if attention_type not in ATTENTION_MODULES:
        raise ValueError(f"Unknown attention type: {attention_type}")

    # 从 kwargs 获取上下文
    seq_len = kwargs.get('seq_len', None)
    input_hw = kwargs.get('input_hw', None)
    H, W = _resolve_hw(seq_len, input_hw)

    # 轻量/自研注意力
    if attention_type == "relative":
        # 防止相对位置索引越界：max_len 至少覆盖序列长度
        max_len = max(512, (seq_len or 512), H * W)
        return RelativePositionSelfAttention(d_model, num_heads, max_len=max_len)
    elif attention_type == "sparse":
        return SparseSelfAttention(d_model, num_heads)
    elif attention_type == "lsh":
        return LSHSelfAttention(d_model, num_heads)

    # 适配不同注意力模块的参数
    if attention_type in ["external"]:
        return ATTENTION_MODULES[attention_type](d_model=d_model, S=8)

    elif attention_type in ["self", "muse", "ufo"]:
        return ATTENTION_MODULES[attention_type](d_model=d_model, d_k=d_model, d_v=d_model, h=num_heads)

    elif attention_type == "simplified_self":
        return ATTENTION_MODULES[attention_type](d_model=d_model, h=num_heads)

    elif attention_type in ["se", "sk", "cbam", "bam"]:
        # 纯通道/空间注意力，通常只需通道数
        return ATTENTION_MODULES[attention_type](channel=d_model, reduction=8)

    elif attention_type == "triplet":
        # TripletAttention 通常不需要显式通道参数
        try:
            return ATTENTION_MODULES[attention_type]()
        except Exception:
            return ATTENTION_MODULES[attention_type](kernel_size=7)

    elif attention_type == "coord":
        # CoordAtt 常见签名为 (inp, oup, reduction)
        try:
            return ATTENTION_MODULES[attention_type](inp=d_model, oup=d_model, reduction=8)
        except Exception:
            try:
                return ATTENTION_MODULES[attention_type](in_channels=d_model, out_channels=d_model, reduction=8)
            except Exception:
                # 回退：无法适配则恒等
                return nn.Identity()

    elif attention_type in ["sge"]:
        g = _safe_groups(d_model, 8)
        return ATTENTION_MODULES[attention_type](groups=g)

    elif attention_type == "eca":
        # ECAAttention 不需要 channel 和 reduction 参数
        return ATTENTION_MODULES[attention_type](kernel_size=3)

    elif attention_type in ["danet"]:
        # 只保证通道数匹配，H/W 由内部处理；不同实现的参数名不同，做鲁棒尝试
        for kwargs_try in (
            {"in_dim": d_model},
            {"in_channels": d_model},
            {"dim": d_model},
            {"d_model": d_model},
            {"channels": d_model},
            {"channel": d_model},
            {"inplanes": d_model},
        ):
            try:
                return ATTENTION_MODULES[attention_type](**kwargs_try)
            except Exception:
                continue
        return nn.Identity()

    elif attention_type in ["psa"]:
        # 跟随模型设备，不要强制 .to('cuda')；参数尽量简单
        try:
            return ATTENTION_MODULES[attention_type](channel=d_model, reduction=8)
        except Exception:
            try:
                return ATTENTION_MODULES[attention_type](in_channels=d_model, reduction=8)
            except Exception:
                return ATTENTION_MODULES[attention_type]()

    elif attention_type in ["shuffle"]:
        g = _safe_groups(d_model, 8)
        try:
            return ATTENTION_MODULES[attention_type](channel=d_model, G=g)
        except Exception:
            # 兼容不同签名
            try:
                return ATTENTION_MODULES[attention_type](in_channels=d_model, G=g)
            except Exception:
                return nn.Identity()

    elif attention_type in ["a2"]:
        # A2(DoubleAttention) 不同实现的构造签名差异较大，逐步尝试
        for ctor in (
            lambda: ATTENTION_MODULES[attention_type](d_model, 128, 128, True),
            lambda: ATTENTION_MODULES[attention_type](in_channels=d_model, c_m=128, c_n=128, reconstruct=True),
            lambda: ATTENTION_MODULES[attention_type](in_channels=d_model, cm=128, cn=128, reconstruct=True),
            lambda: ATTENTION_MODULES[attention_type](in_channels=d_model),
        ):
            try:
                return ctor()
            except Exception:
                continue
        return nn.Identity()

    elif attention_type in ["aft"]:
        # 使用传入的seq_len参数，如果没有则使用默认值
        seq_len_val = seq_len if seq_len is not None else 32
        return ATTENTION_MODULES[attention_type](d_model=d_model, n=seq_len_val)

    elif attention_type in ["outlook"]:
        return ATTENTION_MODULES[attention_type](dim=d_model)

    elif attention_type in ["vip"]:
        # ViP (WeightedPermuteMLP) 要求 seg_dim 与空间尺寸一致：seg_dim == H == W 且 d_model 可被 H 整除
        if H == W and H > 0 and (d_model % H == 0):
            return ATTENTION_MODULES[attention_type](d_model, seg_dim=H)
        else:
            # 若不满足条件则回退为恒等，避免运行时维度错误
            return nn.Identity()

    elif attention_type in ["coatnet"]:
        # 该模块属于整网/骨干，默认不适合作为单层注意力，回退为恒等
        return nn.Identity()

    elif attention_type in ["halo"]:
        # 选择最小的块尺寸，兼容小图
        return ATTENTION_MODULES[attention_type](dim=d_model, block_size=1, halo_size=1)

    elif attention_type in ["polarized"]:
        return ATTENTION_MODULES[attention_type](channel=d_model)

    elif attention_type in ["cot"]:
        return ATTENTION_MODULES[attention_type](dim=d_model, kernel_size=3)

    elif attention_type in ["residual"]:
        return ATTENTION_MODULES[attention_type](channel=d_model, num_class=d_model, la=0.2)

    elif attention_type in ["s2"]:
        # 不同实现可能使用 channels / in_channels / dim 等命名
        for kwargs_try in (
            {"channels": d_model},
            {"in_channels": d_model},
            {"dim": d_model},
            {"channel": d_model},
        ):
            try:
                return ATTENTION_MODULES[attention_type](**kwargs_try)
            except Exception:
                continue
        return nn.Identity()

    elif attention_type in ["gfnet", "mobilevit", "mobilevitv2", "dat", "crossformer", "moa", "crisscross", "axial"]:
        # 这些多为整网/特殊接口，默认回退为恒等，避免不兼容
        return nn.Identity()

    else:
        # 尝试通用构造，不行则回退
        try:
            return ATTENTION_MODULES[attention_type](d_model=d_model, num_heads=num_heads, **kwargs)
        except Exception:
            return nn.Identity()

