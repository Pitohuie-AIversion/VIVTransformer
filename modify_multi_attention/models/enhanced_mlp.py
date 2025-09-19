"""增强的MLP模型，支持稀疏输入预测稠密输出功能（基于PINN简化而来）"""

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
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FourierFeatureMapping(nn.Module):
    """傅里叶特征映射"""
    
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        
        # 随机傅里叶特征
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    
    def forward(self, x):
        # x shape: [..., input_dim]
        x_proj = 2 * math.pi * torch.matmul(x, self.B)
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
        x = self.norm2(x + residual)
        return x

class EnhancedMLP(nn.Module):
    """增强的MLP模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self,
                 input_channels=1,  # 输入通道数
                 output_channels=1,  # 输出通道数
                 hidden_dim=256,
                 num_layers=8,
                 input_resolution=(32, 32),
                 output_resolution=(128, 128),
                 use_fourier_features=True,
                 fourier_mapping_size=256,
                 fourier_scale=10.0,
                 use_residual_blocks=True,
                 dropout=0.1,
                 activation='gelu'):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_fourier_features = use_fourier_features
        self.use_residual_blocks = use_residual_blocks
        
        # 计算输入维度（坐标维度）
        if isinstance(input_resolution, (list, tuple)):
            self.coord_dim = len(input_resolution)
        else:
            self.coord_dim = 1
            self.input_resolution = (input_resolution,)
            
        if isinstance(output_resolution, (list, tuple)):
            self.output_coord_dim = len(output_resolution)
        else:
            self.output_coord_dim = 1
            self.output_resolution = (output_resolution,)
        
        # 激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        # 计算网络输入维度
        # 坐标 + 输入特征
        coord_input_dim = self.coord_dim
        if self.use_fourier_features:
            self.fourier_mapping = FourierFeatureMapping(
                coord_input_dim, fourier_mapping_size, fourier_scale
            )
            network_input_dim = fourier_mapping_size * 2 + max(1, input_channels)  # 确保至少为1
        else:
            network_input_dim = coord_input_dim + max(1, input_channels)  # 确保至少为1
        
        # 网络层
        layers = []
        
        # 输入层
        layers.append(nn.Linear(network_input_dim, hidden_dim))
        layers.append(self.activation)
        
        # 隐藏层
        for i in range(num_layers - 2):
            if self.use_residual_blocks and i % 2 == 0 and i > 0:
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_channels))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _generate_coordinates(self, batch_size, device, resolution=None):
        """生成坐标网格"""
        if resolution is None:
            resolution = self.output_resolution
            
        if len(resolution) == 1:
            # 1D情况
            coords = torch.linspace(0, 1, resolution[0], device=device)
            coords = coords.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        elif len(resolution) == 2:
            # 2D情况
            h, w = resolution
            y_coords = torch.linspace(0, 1, h, device=device)
            x_coords = torch.linspace(0, 1, w, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1)  # [h, w, 2]
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, h, w, 2]
        else:
            raise NotImplementedError(f"Only 1D and 2D coordinates are supported, got {len(resolution)}D")
            
        return coords
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据 [batch, ...] 
        Returns:
            预测输出 [batch, ...]
        """
        original_shape = x.shape
        batch_size = original_shape[0]
        device = x.device
        
        # 生成目标分辨率的坐标网格
        coords = self._generate_coordinates(batch_size, device)
        
        # 处理输入数据 - 修复版
        if len(original_shape) == 2:
            # [batch, features] -> 扩展到目标分辨率
            if len(self.output_resolution) == 1:
                x_expanded = x.unsqueeze(1).expand(-1, self.output_resolution[0], -1)
            elif len(self.output_resolution) == 2:
                x_expanded = x.unsqueeze(1).unsqueeze(1).expand(-1, *self.output_resolution, -1)
        elif len(original_shape) == 3:
            # [batch, spatial, features] 或 [batch, height, width]
            if original_shape[-1] == 1 or original_shape[-1] == self.input_channels:
                # 有特征维度的情况
                if len(self.output_resolution) == 1:
                    x_expanded = F.interpolate(x.transpose(1, 2), size=self.output_resolution[0], mode='linear', align_corners=False).transpose(1, 2)
                else:
                    # 3D输入 [batch, height, width] -> 4D输出 [batch, height, width, channels]
                    x_expanded = x.unsqueeze(-1)  # [batch, height, width, 1]
                    x_expanded = F.interpolate(x_expanded.permute(0, 3, 1, 2), size=self.output_resolution, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            else:
                # 没有特征维度，添加特征维度
                x = x.unsqueeze(-1)  # [batch, height, width, 1]
                x_expanded = F.interpolate(x.permute(0, 3, 1, 2), size=self.output_resolution, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        elif len(original_shape) == 4:
            # [batch, channels, height, width] 或 [batch, height, width, channels]
            if original_shape[1] == 1 or original_shape[1] == self.input_channels:
                # [batch, channels, height, width] 格式
                x_expanded = F.interpolate(x, size=self.output_resolution, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            else:
                # [batch, height, width, channels] 格式
                x_expanded = F.interpolate(x.permute(0, 3, 1, 2), size=self.output_resolution, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        # 合并坐标和特征
        if len(self.output_resolution) == 1:
            # 1D情况
            coords_flat = coords.reshape(-1, self.coord_dim)  # [batch * length, coord_dim]
            features_flat = x_expanded.reshape(-1, self.input_channels)  # [batch * length, input_channels]
        else:
            # 2D情况
            coords_flat = coords.reshape(-1, self.coord_dim)  # [batch * h * w, coord_dim]
            features_flat = x_expanded.reshape(-1, self.input_channels)  # [batch * h * w, input_channels]
        
        # 傅里叶特征映射（仅对坐标）
        if self.use_fourier_features:
            coords_flat = self.fourier_mapping(coords_flat)
        
        # 合并坐标特征和输入特征
        if self.input_channels == 0 or features_flat.shape[-1] == 0:
            # 如果没有输入特征，只使用坐标
            network_input = coords_flat
        else:
            # 确保维度匹配
            if coords_flat.shape[0] != features_flat.shape[0]:
                # 如果坐标和特征的batch*spatial维度不匹配，调整特征维度
                target_size = coords_flat.shape[0]
                features_flat = features_flat.repeat(target_size // features_flat.shape[0], 1)
            
            network_input = torch.cat([coords_flat, features_flat], dim=-1)
        
        # 网络前向传播
        output_flat = self.network(network_input)
        
        # 重塑输出
        if len(self.output_resolution) == 1:
            output = output_flat.reshape(batch_size, self.output_resolution[0], self.output_channels)
        else:
            output = output_flat.reshape(batch_size, *self.output_resolution, self.output_channels)
        
        return output

class EnhancedMLP1d(EnhancedMLP):
    """1D增强MLP模型"""
    
    def __init__(self, 
                 input_channels=1,
                 output_channels=1,
                 hidden_dim=256,
                 num_layers=8,
                 input_resolution=32,
                 output_resolution=128,
                 **kwargs):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            **kwargs
        )

class EnhancedMLP2d(EnhancedMLP):
    """2D增强MLP模型"""
    
    def __init__(self, 
                 input_channels=1,
                 output_channels=1,
                 hidden_dim=256,
                 num_layers=8,
                 input_resolution=(32, 32),
                 output_resolution=(128, 128),
                 **kwargs):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            **kwargs
        )

class SparseToDeseAdapterMLP(nn.Module):
    """MLP模型的稀疏到稠密适配器"""
    
    def __init__(self, 
                 mlp_model: nn.Module,
                 input_resolution: Tuple[int, ...],
                 output_resolution: Tuple[int, ...]):
        super().__init__()
        self.mlp_model = mlp_model
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        
    def forward(self, sparse_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_input: 稀疏输入张量
        Returns:
            稠密输出张量
        """
        dense_output = self.mlp_model(sparse_input)
        return dense_output

# 工厂函数
def create_enhanced_mlp1d(input_channels=1,
                          output_channels=1,
                          hidden_dim=256,
                          num_layers=8,
                          input_resolution=32,
                          output_resolution=128,
                          **kwargs) -> EnhancedMLP1d:
    """创建增强的MLP1d模型"""
    return EnhancedMLP1d(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        **kwargs
    )

def create_enhanced_mlp2d(input_channels=1,
                          output_channels=1,
                          hidden_dim=256,
                          num_layers=8,
                          input_resolution=(32, 32),
                          output_resolution=(128, 128),
                          **kwargs) -> EnhancedMLP2d:
    """创建增强的MLP2d模型"""
    return EnhancedMLP2d(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        **kwargs
    )

if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试MLP1d
    print("测试EnhancedMLP1d...")
    model_1d = create_enhanced_mlp1d(
        input_channels=1,
        output_channels=1,
        input_resolution=32,
        output_resolution=128
    ).to(device)
    
    x_1d = torch.randn(4, 32, 1).to(device)
    output_1d = model_1d(x_1d)
    print(f"MLP1d输入形状: {x_1d.shape}, 输出形状: {output_1d.shape}")
    
    # 测试MLP2d
    print("\n测试EnhancedMLP2d...")
    model_2d = create_enhanced_mlp2d(
        input_channels=1,
        output_channels=1,
        input_resolution=(32, 32),
        output_resolution=(128, 128)
    ).to(device)
    
    x_2d = torch.randn(4, 32, 32, 1).to(device)
    output_2d = model_2d(x_2d)
    print(f"MLP2d输入形状: {x_2d.shape}, 输出形状: {output_2d.shape}")
    
    # 测试不同输入格式
    print("\n测试不同输入格式...")
    
    # 测试批量特征输入
    x_batch = torch.randn(4, 1).to(device)
    output_batch = model_1d(x_batch)
    print(f"批量输入形状: {x_batch.shape}, 输出形状: {output_batch.shape}")
    
    # 测试2D批量特征输入
    x_batch_2d = torch.randn(4, 1).to(device)
    output_batch_2d = model_2d(x_batch_2d)
    print(f"2D批量输入形状: {x_batch_2d.shape}, 输出形状: {output_batch_2d.shape}")
    
    print("\n所有测试通过！")