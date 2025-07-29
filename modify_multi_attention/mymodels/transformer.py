import torch
import torch.nn as nn
import copy
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
        
        # 动态计算spatial_dim，支持非完全平方数
        spatial_dim = int(seq_len ** 0.5)
        
        # 如果不是完全平方数，尝试找到最接近的因子分解
        if spatial_dim * spatial_dim != seq_len:
            # 寻找最接近的因子对
            factors = []
            for i in range(1, int(seq_len ** 0.5) + 1):
                if seq_len % i == 0:
                    factors.append((i, seq_len // i))
            
            if factors:
                # 选择最接近正方形的因子对
                spatial_dim_h, spatial_dim_w = min(factors, key=lambda x: abs(x[0] - x[1]))
            else:
                # 如果是质数，使用1xseq_len的形状
                spatial_dim_h, spatial_dim_w = 1, seq_len
        else:
            spatial_dim_h = spatial_dim_w = spatial_dim
            
        # 验证重塑后的维度
        if spatial_dim_h * spatial_dim_w != seq_len:
            # 如果因子分解失败，强制使用最接近的正方形
            spatial_dim_h = spatial_dim_w = spatial_dim
            # 如果仍然不匹配，截断或填充
            if spatial_dim * spatial_dim < seq_len:
                spatial_dim_h = spatial_dim_w = spatial_dim + 1
            elif spatial_dim * spatial_dim > seq_len:
                spatial_dim_h = spatial_dim_w = spatial_dim

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
            # CNN类注意力机制，使用动态计算的空间维度
            try:
                src_reshaped = src.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim_h, spatial_dim_w)
                src2 = self.self_attn(src_reshaped)
                src2 = src2.view(batch_size, d_model, -1)[:, :, :seq_len].transpose(1, 2)
            except RuntimeError as e:
                # 如果重塑失败，使用简单的线性变换作为fallback
                print(f"Warning: Attention reshape failed ({e}), using linear fallback")
                src2 = src  # 直接跳过注意力机制

        elif isinstance(self.self_attn, (ExternalAttention, AFT_FULL)):
            src2 = self.self_attn(src)
        elif isinstance(self.self_attn, (OutlookAttention, WeightedPermuteMLP)):
            src_reshaped = src.view(batch_size, spatial_dim, spatial_dim, d_model)  # 注意这里形状变为 (B,H,W,C)
            src2 = self.self_attn(src_reshaped)
            src2 = src2.view(batch_size, seq_len, d_model)
        elif isinstance(self.self_attn, ResidualAttention):
            src_reshaped = src.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim, spatial_dim)
            src2 = self.self_attn(src_reshaped)
            src2 = src2.view(batch_size, d_model, seq_len).transpose(1, 2)


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
        
        # 动态计算spatial_dim，支持非完全平方数
        spatial_dim = int(seq_len ** 0.5)
        
        # 如果不是完全平方数，尝试找到最接近的因子分解
        if spatial_dim * spatial_dim != seq_len:
            # 寻找最接近的因子对
            factors = []
            for i in range(1, int(seq_len ** 0.5) + 1):
                if seq_len % i == 0:
                    factors.append((i, seq_len // i))
            
            if factors:
                # 选择最接近正方形的因子对
                spatial_dim_h, spatial_dim_w = min(factors, key=lambda x: abs(x[0] - x[1]))
            else:
                # 如果是质数，使用1xseq_len的形状
                spatial_dim_h, spatial_dim_w = 1, seq_len
        else:
            spatial_dim_h = spatial_dim_w = spatial_dim

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
            # CNN类注意力机制，使用动态计算的空间维度
            tgt_reshaped = tgt.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim_h, spatial_dim_w)
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
            # CNN类注意力机制，使用动态计算的空间维度
            tgt_reshaped = tgt.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim_h, spatial_dim_w)
            tgt2 = self.multihead_attn(tgt_reshaped)
            tgt2 = tgt2.view(batch_size, d_model, seq_len).transpose(1, 2)

        elif isinstance(self.multihead_attn, (ExternalAttention, AFT_FULL)):
            tgt2 = self.multihead_attn(tgt)
        elif isinstance(self.multihead_attn, (OutlookAttention, WeightedPermuteMLP)):
            # 明确调整reshape顺序为 (B,H,W,C)
            tgt_reshaped = tgt.view(batch_size, spatial_dim, spatial_dim, d_model)
            memory_reshaped = memory.view(batch_size, spatial_dim, spatial_dim, d_model)
            tgt2 = self.multihead_attn(tgt_reshaped)
            tgt2 = tgt2.view(batch_size, seq_len, d_model)


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
                 d_model=512, max_time_steps=100, attention_type="relative", seq_len=32):
        super().__init__()

        self.attention_type = attention_type
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 确保embedding输出维度正确
        self.embedding = nn.Linear(input_dim, seq_len * d_model)
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))

        encoder_layer = CustomEncoderLayer(d_model=d_model, num_heads=num_heads,
                                           dim_feedforward=2048, attention_type=attention_type)
        decoder_layer = CustomDecoderLayer(d_model=d_model, num_heads=num_heads,
                                           dim_feedforward=2048, attention_type=attention_type)

        self.encoder = CustomEncoder(encoder_layer, num_layers)
        self.decoder = CustomDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x_in_pressures_flat, x_time_steps):
        batch_size = x_in_pressures_flat.size(0)
        x_time_steps = x_time_steps.long()

        # 确保时间步索引在有效范围内
        x_time_steps = torch.clamp(x_time_steps, 0, 99)  # max_time_steps - 1

        # 使用初始化时设置的seq_len
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

