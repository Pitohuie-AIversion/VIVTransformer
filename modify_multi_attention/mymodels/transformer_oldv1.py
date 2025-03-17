import torch
import torch.nn as nn
from components.attention import RelativePositionSelfAttention
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 替换为任意一种新的自注意力机制
        # 示例：使用相对位置注意力
        self.self_attn = RelativePositionSelfAttention(d_model, num_heads)

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
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 自注意力层（例如相对位置注意力）
        self.self_attn = RelativePositionSelfAttention(d_model, num_heads)

        # 交叉注意力层（同样可以替换为其他类型）
        self.multihead_attn = RelativePositionSelfAttention(d_model, num_heads)

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
        tgt2 = self.self_attn(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt)  # 假设memory也经过相同处理
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

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
                 d_model=512, max_time_steps=100):
        super().__init__()

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, d_model)

        # 时间步嵌入层
        self.time_step_embedding = nn.Embedding(max_time_steps, d_model)

        # 位置编码（可选）
        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model))

        # 自定义编码器和解码器
        encoder_layer = CustomEncoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dim_feedforward=2048)
        decoder_layer = CustomDecoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dim_feedforward=2048)

        self.encoder = CustomEncoder(encoder_layer=encoder_layer,
                                     num_layers=num_layers)

        self.decoder = CustomDecoder(decoder_layer=decoder_layer,
                                     num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x_in_pressures_flat, x_time_steps):
        x_time_steps = x_time_steps.long()
        # 输入嵌入和时间步嵌入
        x_embedded = self.embedding(x_in_pressures_flat) + \
                     self.time_step_embedding(x_time_steps) + \
                     self.positional_encoding

        # 编码器处理
        encoder_output = self.encoder(x_embedded.unsqueeze(1))  # 保持维度 [batch, seq_len=1, d_model]

        # 解码器处理 (不做squeeze)
        decoder_output = self.decoder(encoder_output, encoder_output)

        # 最终输出
        out_pressure_flat_pred = self.fc_out(decoder_output.mean(dim=1))

        return out_pressure_flat_pred
