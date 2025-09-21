"""
修复版增强MLP模型
解决稀疏到稠密重建中的维度不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:x.size(1), :].unsqueeze(0).expand(x.size(0), -1, -1)

class FourierFeatureMapping(nn.Module):
    """傅里叶特征映射"""
    
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        # 固定的随机投影矩阵
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    
    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual
        return self.norm2(x)

class FixedEnhancedMLP(nn.Module):
    """修复版增强MLP模型 - 专门处理稀疏到稠密重建"""
    
    def __init__(self,
                 input_dim=1024,           # 输入维度
                 output_dim=16384,         # 输出维度  
                 hidden_dim=512,           # 隐藏层维度
                 num_layers=6,             # 网络层数
                 use_fourier_features=True, # 是否使用傅里叶特征
                 fourier_mapping_size=256,  # 傅里叶映射大小
                 fourier_scale=10.0,       # 傅里叶缩放因子
                 use_residual_blocks=True,  # 是否使用残差块
                 dropout=0.1,              # Dropout率
                 activation='gelu'):       # 激活函数
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_fourier_features = use_fourier_features
        self.use_residual_blocks = use_residual_blocks
        
        # 计算输入和输出的空间分辨率
        self.input_resolution = int(math.sqrt(input_dim))  # 32x32 = 1024
        self.output_resolution = int(math.sqrt(output_dim))  # 128x128 = 16384
        
        # 坐标维度 (x, y)
        self.coord_dim = 2
        
        # 傅里叶特征映射
        if self.use_fourier_features:
            self.fourier_mapping = FourierFeatureMapping(
                self.coord_dim, fourier_mapping_size, fourier_scale
            )
            coord_feature_dim = fourier_mapping_size * 2
        else:
            coord_feature_dim = self.coord_dim
        
        # 输入特征处理
        self.input_projection = nn.Linear(1, hidden_dim // 4)  # 每个像素值投影
        
        # 网络输入维度：坐标特征 + 输入特征
        network_input_dim = coord_feature_dim + hidden_dim // 4
        
        # 构建网络
        layers = []
        
        # 输入层
        layers.append(nn.Linear(network_input_dim, hidden_dim))
        layers.append(self._get_activation(activation))
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for i in range(num_layers - 2):
            if use_residual_blocks and i > 0:
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self._get_activation(activation))
                layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, 1))  # 输出单个像素值
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _get_activation(self, activation):
        """获取激活函数"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _generate_coordinates(self, batch_size, device):
        """生成输出分辨率的坐标网格"""
        # 生成归一化坐标 [-1, 1]
        x = torch.linspace(-1, 1, self.output_resolution, device=device)
        y = torch.linspace(-1, 1, self.output_resolution, device=device)
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # 展平并组合坐标 [H*W, 2]
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        
        # 扩展到批次维度 [B, H*W, 2]
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
        
        return coords
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据 [batch, input_dim] 
        Returns:
            预测输出 [batch, output_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 确保输入是2D张量 [batch, input_dim]
        if len(x.shape) != 2:
            raise ValueError(f"期望输入形状为 [batch, input_dim]，实际得到 {x.shape}")
        
        if x.shape[1] != self.input_dim:
            raise ValueError(f"期望输入维度为 {self.input_dim}，实际得到 {x.shape[1]}")
        
        # 1. 将输入重塑为2D图像 [batch, H_in, W_in]
        input_h = input_w = self.input_resolution
        x_2d = x.view(batch_size, input_h, input_w)
        
        # 2. 上采样到输出分辨率 [batch, H_out, W_out]
        x_upsampled = F.interpolate(
            x_2d.unsqueeze(1),  # [batch, 1, H_in, W_in]
            size=(self.output_resolution, self.output_resolution),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [batch, H_out, W_out]
        
        # 3. 生成输出坐标网格 [batch, H_out*W_out, 2]
        coords = self._generate_coordinates(batch_size, device)
        
        # 4. 处理输入特征
        # 将上采样后的图像展平 [batch, H_out*W_out]
        x_flat = x_upsampled.view(batch_size, -1)
        
        # 投影到特征空间 [batch, H_out*W_out, feature_dim]
        input_features = self.input_projection(x_flat.unsqueeze(-1))
        
        # 5. 处理坐标特征
        if self.use_fourier_features:
            # 应用傅里叶特征映射 [batch, H_out*W_out, fourier_dim]
            coord_features = self.fourier_mapping(coords)
        else:
            coord_features = coords
        
        # 6. 合并坐标和输入特征 [batch, H_out*W_out, total_dim]
        combined_features = torch.cat([coord_features, input_features], dim=-1)
        
        # 7. 网络前向传播
        # 重塑为 [batch*H_out*W_out, total_dim] 以便并行处理
        combined_flat = combined_features.view(-1, combined_features.shape[-1])
        
        # 通过网络 [batch*H_out*W_out, 1]
        output_flat = self.network(combined_flat)
        
        # 8. 重塑输出 [batch, H_out*W_out] -> [batch, output_dim]
        output = output_flat.view(batch_size, -1)
        
        return output

class FixedEnhancedMLP1d(FixedEnhancedMLP):
    """1D版本的修复增强MLP（实际上处理的是2D数据的1D表示）"""
    
    def __init__(self, 
                 input_channels=1,
                 output_channels=1,
                 hidden_dim=512,
                 num_layers=6,
                 input_resolution=32,
                 output_resolution=128,
                 **kwargs):
        # 计算实际的输入输出维度
        input_dim = input_resolution * input_resolution * input_channels
        output_dim = output_resolution * output_resolution * output_channels
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs
        )

# 工厂函数
def create_fixed_enhanced_mlp1d(input_channels=1,
                                output_channels=1,
                                hidden_dim=512,
                                num_layers=6,
                                input_resolution=32,
                                output_resolution=128,
                                **kwargs) -> FixedEnhancedMLP1d:
    """创建修复版增强MLP1d模型"""
    return FixedEnhancedMLP1d(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        **kwargs
    )

def create_fixed_enhanced_mlp(input_dim=1024,
                              output_dim=16384,
                              hidden_dim=512,
                              num_layers=6,
                              **kwargs) -> FixedEnhancedMLP:
    """创建修复版增强MLP模型"""
    return FixedEnhancedMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs
    )

if __name__ == "__main__":
    # 测试修复版MLP模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("测试修复版FixedEnhancedMLP...")
    model = create_fixed_enhanced_mlp(
        input_dim=1024,
        output_dim=16384,
        hidden_dim=512,
        num_layers=6
    ).to(device)
    
    # 测试输入：[batch, 1024]
    x = torch.randn(4, 1024).to(device)
    output = model(x)
    print(f"输入形状: {x.shape}, 输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n测试修复版FixedEnhancedMLP1d...")
    model_1d = create_fixed_enhanced_mlp1d(
        input_channels=1,
        output_channels=1,
        input_resolution=32,
        output_resolution=128,
        hidden_dim=512,
        num_layers=6
    ).to(device)
    
    # 测试输入：[batch, 1024]
    x_1d = torch.randn(4, 1024).to(device)
    output_1d = model_1d(x_1d)
    print(f"1D输入形状: {x_1d.shape}, 输出形状: {output_1d.shape}")
    print(f"1D参数数量: {sum(p.numel() for p in model_1d.parameters()):,}")
    
    print("\n所有测试通过！")