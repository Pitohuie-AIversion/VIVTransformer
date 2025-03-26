import torch.nn as nn
from modify_multi_attention.mymodels.components.attention import RelativePositionSelfAttention, SparseSelfAttention, LSHSelfAttention

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

def get_attention_module(attention_type, d_model=512, num_heads=8, **kwargs):
    """
    根据 attention_type 返回对应的注意力模块，适配不同注意力模块的参数
    """
    if attention_type not in ATTENTION_MODULES:
        raise ValueError(f"Unknown attention type: {attention_type}")
    if attention_type == "relative":
        return RelativePositionSelfAttention(d_model, num_heads)
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

    elif attention_type in ["se", "sk", "cbam", "bam", "triplet", "coord"]:
        return ATTENTION_MODULES[attention_type](channel=d_model, reduction=8)
    if attention_type in ["sge"]:
        return ATTENTION_MODULES[attention_type](groups=8)
    elif attention_type == "eca":
        # ECAAttention 不需要 channel 和 reduction 参数
        return ATTENTION_MODULES[attention_type](kernel_size=3)

    elif attention_type in ["danet"]:
        return ATTENTION_MODULES[attention_type](d_model=d_model, kernel_size=3, H=7, W=7)

    elif attention_type in ["psa"]:
        return ATTENTION_MODULES[attention_type](channel=d_model, reduction=8).to('cuda')

    elif attention_type in ["shuffle"]:
        return ATTENTION_MODULES[attention_type](channel=d_model, G=8)

    elif attention_type in ["a2"]:
        return ATTENTION_MODULES[attention_type](d_model, 128, 128, True)

    elif attention_type in ["aft"]:
        return ATTENTION_MODULES[attention_type](d_model=d_model, n=49)

    elif attention_type in ["outlook"]:
        return ATTENTION_MODULES[attention_type](dim=d_model)

    elif attention_type in ["vip"]:
        return ATTENTION_MODULES[attention_type](d_model, seg_dim=8)

    elif attention_type in ["coatnet"]:
        return ATTENTION_MODULES[attention_type](in_ch=d_model, image_size=spatial_dim)

    elif attention_type in ["halo"]:
        return ATTENTION_MODULES[attention_type](dim=d_model, block_size=1, halo_size=1)

    elif attention_type in ["polarized"]:
        return ATTENTION_MODULES[attention_type](channel=d_model)

    elif attention_type in ["cot"]:
        return ATTENTION_MODULES[attention_type](dim=d_model, kernel_size=3)

    elif attention_type in ["residual"]:
        return ATTENTION_MODULES[attention_type](channel=d_model, num_class=d_model, la=0.2)

    elif attention_type in ["s2"]:
        return ATTENTION_MODULES[attention_type](channels=d_model)

    elif attention_type in ["gfnet"]:
        return ATTENTION_MODULES[attention_type](embed_dim=384, img_size=224, patch_size=16, num_classes=1000)

    elif attention_type in ["mobilevit"]:
        return ATTENTION_MODULES[attention_type]()

    elif attention_type in ["parnet"]:
        return ATTENTION_MODULES[attention_type](channel=d_model)

    elif attention_type in ["mobilevitv2"]:
        return ATTENTION_MODULES[attention_type](d_model=d_model)

    elif attention_type in ["dat"]:
        return ATTENTION_MODULES[attention_type](
            img_size=224, patch_size=4, num_classes=1000, expansion=4,
            dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
            stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
            heads=[3, 6, 12, 24], window_sizes=[7, 7, 7, 7],
            groups=[-1, -1, 3, 6], use_pes=[False, False, True, True],
            dwc_pes=[False, False, False, False], strides=[-1, -1, 1, 1],
            sr_ratios=[-1, -1, -1, -1], offset_range_factor=[-1, -1, 2, 2],
            no_offs=[False, False, False, False], fixed_pes=[False, False, False, False],
            use_dwc_mlps=[False, False, False, False], use_conv_patches=False,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2
        )

    elif attention_type in ["crossformer"]:
        return ATTENTION_MODULES[attention_type](
            img_size=224, patch_size=[4, 8, 16, 32], in_chans=3,
            num_classes=1000, embed_dim=48, depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24], group_size=[7, 7, 7, 7],
            mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.0,
            drop_path_rate=0.1, ape=False, patch_norm=True,
            use_checkpoint=False, merge_size=[[2, 4], [2, 4], [2, 4]]
        )

    elif attention_type in ["moa"]:
        return ATTENTION_MODULES[attention_type](
            img_size=224, patch_size=4, in_chans=3, num_classes=1000,
            embed_dim=96, depths=[2, 2, 6], num_heads=[3, 6, 12],
            window_size=14, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0.0, drop_path_rate=0.1, ape=False, patch_norm=True,
            use_checkpoint=False
        )

    elif attention_type in ["crisscross"]:
        return ATTENTION_MODULES[attention_type](dim=128, depth=12, reversible=True)

    else:
        return ATTENTION_MODULES[attention_type](d_model=d_model, num_heads=num_heads)

