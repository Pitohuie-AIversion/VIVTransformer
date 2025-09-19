
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的FNO模型
解决矩阵维度不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SpectralConv2d_fast(nn.Module):
    """快速谱卷积层"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FixedEnhancedFNO2d(nn.Module):
    """修复后的增强FNO2d模型"""
    
    def __init__(self, 
                 num_channels=1, 
                 modes1=12, 
                 modes2=12, 
                 width=20, 
                 input_resolution=(32, 32),
                 output_resolution=(128, 128),
                 use_upsampling=True):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_upsampling = use_upsampling
        self.num_channels = num_channels
        
        # 输入投影层 - 修复维度匹配问题
        # 输入: [batch, h, w, channels + 2] -> [batch, h, w, width]
        self.fc0 = nn.Linear(num_channels + 2, self.width)
        
        # Fourier层
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        
        # 卷积层
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # 上采样层（如果需要）
        if self.use_upsampling and (output_resolution[0] > input_resolution[0] or 
                                   output_resolution[1] > input_resolution[1]):
            # 计算上采样倍数
            scale_h = output_resolution[0] / input_resolution[0]
            scale_w = output_resolution[1] / input_resolution[1]
            
            if scale_h == 4 and scale_w == 4:  # 4x上采样
                self.upsampler = nn.Sequential(
                    nn.ConvTranspose2d(self.width, self.width, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(self.width, self.width, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.GELU()
                )
            elif scale_h == 2 and scale_w == 2:  # 2x上采样
                self.upsampler = nn.Sequential(
                    nn.ConvTranspose2d(self.width, self.width, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.GELU()
                )
            else:
                # 使用自适应上采样
                self.upsampler = nn.Sequential(
                    nn.ConvTranspose2d(self.width, self.width, 
                                     kernel_size=3, stride=1, padding=1),
                    nn.GELU()
                )
        else:
            self.upsampler = None
        
        # 输出投影层 - 修复维度匹配
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)
        
    def forward(self, x, grid=None):
        # 处理不同的输入格式
        if len(x.shape) == 4 and x.shape[1] == self.num_channels:
            # 输入格式: [batch, channels, h, w] -> [batch, h, w, channels]
            x = x.permute(0, 2, 3, 1)
        
        batch_size, h, w = x.shape[0], x.shape[1], x.shape[2]
        
        # 如果没有提供grid，创建默认的位置编码
        if grid is None:
            # 创建2D网格
            y_coords = torch.linspace(0, 1, h, device=x.device)
            x_coords = torch.linspace(0, 1, w, device=x.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
            grid = grid.expand(batch_size, -1, -1, -1)
        
        # 连接输入和位置编码
        x = torch.cat((x, grid), dim=-1)  # [batch, height, width, channels+2]
        
        # 输入投影
        x = self.fc0(x)  # [batch, height, width, width]
        x = x.permute(0, 3, 1, 2)  # [batch, width, height, width]
        
        # 填充
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Fourier层
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        # 去除填充
        x = x[..., :-self.padding, :-self.padding]
        
        # 上采样到目标分辨率
        if self.upsampler is not None:
            x = self.upsampler(x)
            # 调整到精确的输出分辨率
            if x.shape[-2:] != self.output_resolution:
                x = F.interpolate(x, size=self.output_resolution, mode='bilinear', align_corners=False)
        elif self.output_resolution != self.input_resolution:
            # 使用插值调整分辨率
            x = F.interpolate(x, size=self.output_resolution, mode='bilinear', align_corners=False)
        
        # 输出投影
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        # 返回标准格式 [batch, channels, h, w]
        if len(x.shape) == 4 and x.shape[-1] == self.num_channels:
            x = x.permute(0, 3, 1, 2)
        
        return x

class PDEBenchFNOWrapper(nn.Module):
    """FNO模型的PDEBench兼容性包装器"""
    
    def __init__(self, fno_model):
        super().__init__()
        self.fno_model = fno_model
    
    def forward(self, x, time_steps=None):
        """
        Args:
            x: 输入数据 [batch, channels, h, w]
            time_steps: 时间步信息（可选）
        Returns:
            输出数据 [batch, channels, h, w]
        """
        return self.fno_model(x)

def create_fixed_fno(num_channels=1, input_resolution=(32, 32), 
                    output_resolution=(128, 128)):
    """创建修复后的FNO模型"""
    base_model = FixedEnhancedFNO2d(
        num_channels=num_channels,
        input_resolution=input_resolution,
        output_resolution=output_resolution
    )
    
    return PDEBenchFNOWrapper(base_model)
