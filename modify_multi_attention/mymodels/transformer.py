import torch
import torch.nn as nn
import copy
from modify_multi_attention.mymodels.components.attention_factory import get_attention_module  # 动态获取注意力模块

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

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, attention_type="relative"):
        super().__init__()

        # 选择注意力机制
        self.self_attn = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads)

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
        spatial_dim = int(seq_len ** 0.5)

        # ---- Self-Attention 适配 ----
        if isinstance(self.self_attn, ExternalAttention):
            src2 = self.self_attn(src)

        elif isinstance(self.self_attn, (SEAttention, SKAttention, CBAMBlock, BAMBlock,
                                         ECAAttention, ShuffleAttention, SpatialGroupEnhance,
                                         ResidualAttention, S2Attention, TripletAttention,
                                         CoordAtt, PSA, DAModule, CoTAttention,
                                         SequentialPolarizedSelfAttention, SequentialPolarizedSelfAttention,
                                         OutlookAttention, WeightedPermuteMLP, CoAtNet,
                                         HaloAttention, DoubleAttention, ParNetAttention, )):
            # CNN类注意力机制
            if spatial_dim * spatial_dim != seq_len:
                raise ValueError("Sequence length cannot form square spatial dimensions for CNN attention.")
            src_reshaped = src.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim, spatial_dim)
            src2 = self.self_attn(src_reshaped)
            src2 = src2.view(batch_size, d_model, seq_len).transpose(1, 2)

        elif isinstance(self.self_attn, (ExternalAttention, AFT_FULL)):
            src2 = self.self_attn(src)

        # 默认Transformer类机制（QKV）
        else:
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
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, attention_type="relative"):
        super().__init__()

        # 选择注意力机制
        self.self_attn = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads)
        self.multihead_attn = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads)

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
        spatial_dim = int(seq_len ** 0.5)

        # ---- Self-Attention 适配 ----
        if isinstance(self.self_attn, ExternalAttention):
            tgt2 = self.self_attn(tgt)

        elif isinstance(self.self_attn, (SEAttention, SKAttention, CBAMBlock, BAMBlock,
                                         ECAAttention, ShuffleAttention, SpatialGroupEnhance,
                                         ResidualAttention, S2Attention, TripletAttention,
                                         CoordAtt, PSA, DAModule, CoTAttention,
                                         SequentialPolarizedSelfAttention, SequentialPolarizedSelfAttention,
                                         OutlookAttention, WeightedPermuteMLP, CoAtNet,
                                         HaloAttention, DoubleAttention, ParNetAttention, )):
            # CNN类注意力机制
            tgt_reshaped = tgt.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim, spatial_dim)
            tgt2 = self.self_attn(tgt_reshaped)
            tgt2 = tgt2.view(batch_size, d_model, seq_len).transpose(1, 2)

        # ExternalAttention、AFT等特殊机制单独适配
        elif isinstance(self.self_attn, (ExternalAttention, AFT_FULL)):
            tgt2 = self.self_attn(tgt)

        # 默认Transformer类机制（QKV）
        else:
            tgt2 = self.self_attn(tgt, tgt, tgt)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 交叉注意力（multihead_attn）
        if isinstance(self.multihead_attn, (SEAttention, SKAttention, CBAMBlock, BAMBlock,
                                            ECAAttention, ShuffleAttention, SpatialGroupEnhance,
                                            ResidualAttention, S2Attention, TripletAttention,
                                            CoordAtt, PSA, DAModule, CoTAttention,
                                            SequentialPolarizedSelfAttention, SequentialPolarizedSelfAttention,
                                            OutlookAttention, WeightedPermuteMLP, CoAtNet,
                                            HaloAttention, DoubleAttention, ParNetAttention)):
            # CNN类注意力机制需要reshape到4维
            if spatial_dim * spatial_dim != seq_len:
                raise ValueError("Sequence length cannot form square spatial dimensions for CNN attention.")
            tgt_reshaped = tgt.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim, spatial_dim)
            tgt2 = self.multihead_attn(tgt_reshaped)
            tgt2 = tgt2.view(batch_size, d_model, seq_len).transpose(1, 2)

        elif isinstance(self.multihead_attn, (ExternalAttention, AFT_FULL)):
            tgt2 = self.multihead_attn(tgt)

        # 默认使用标准Transformer交叉注意力(Q,K,V)
        else:
            tgt2 = self.multihead_attn(tgt, memory, memory)

        # 残差连接 + LayerNorm
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
                 d_model=512, max_time_steps=100, attention_type="relative"):
        super().__init__()

        # 记录当前使用的注意力机制
        self.attention_type = attention_type

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, d_model)
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)

        # 位置编码（可选）
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))

        # 选择注意力机制
        encoder_layer = CustomEncoderLayer(d_model=d_model, num_heads=num_heads,
                                           dim_feedforward=2048, attention_type=attention_type)
        decoder_layer = CustomDecoderLayer(d_model=d_model, num_heads=num_heads,
                                           dim_feedforward=2048, attention_type=attention_type)

        self.encoder = CustomEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.decoder = CustomDecoder(decoder_layer=decoder_layer, num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x_in_pressures_flat, x_time_steps):
        # 直接在 `forward` 里打印当前使用的注意力机制
        # print(f"\033[92m当前使用的注意力机制: {self.attention_type}\033[0m")  # 绿色高亮输出
        x_time_steps = x_time_steps.long()

        # 输入嵌入和时间步嵌入
        x_embedded = self.embedding(x_in_pressures_flat) + \
                     self.time_step_embedding(x_time_steps) + \
                     self.positional_encoding

        # 编码器处理
        encoder_output = self.encoder(x_embedded.unsqueeze(1))  # 维度 [batch, seq_len=1, d_model]

        # 解码器处理
        decoder_output = self.decoder(encoder_output, encoder_output)

        # 最终输出
        out_pressure_flat_pred = self.fc_out(decoder_output.mean(dim=1))

        return out_pressure_flat_pred
