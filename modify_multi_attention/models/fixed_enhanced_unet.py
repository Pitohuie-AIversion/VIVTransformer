"""
修复版增强U-Net模型
解决稀疏到稠密重建中的维度不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, List

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class FixedEnhancedUNet(nn.Module):
    """修复版增强U-Net模型 - 专门处理稀疏到稠密重建"""
    
    def __init__(self, 
                 input_dim=1024,           # 输入维度
                 output_dim=16384,         # 输出维度
                 in_channels=1,            # 输入通道数
                 out_channels=1,           # 输出通道数
                 init_features=64,         # 初始特征数
                 bilinear=False):          # 是否使用双线性插值
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 计算输入和输出的空间分辨率
        self.input_resolution = int(math.sqrt(input_dim))  # 32x32 = 1024
        self.output_resolution = int(math.sqrt(output_dim))  # 128x128 = 16384
        
        features = init_features
        
        # 输入层
        self.inc = DoubleConv(in_channels, features)
        
        # 编码器
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)
        
        # 解码器
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)
        
        # 输出层
        self.outc = OutConv(features, out_channels)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入数据 [batch, input_dim]
        Returns:
            预测输出 [batch, output_dim]
        """
        batch_size = x.shape[0]
        
        # 确保输入是2D张量 [batch, input_dim]
        if len(x.shape) != 2:
            raise ValueError(f"期望输入形状为 [batch, input_dim]，实际得到 {x.shape}")
        
        if x.shape[1] != self.input_dim:
            raise ValueError(f"期望输入维度为 {self.input_dim}，实际得到 {x.shape[1]}")
        
        # 1. 将输入重塑为2D图像 [batch, channels, height, width]
        x = x.view(batch_size, self.in_channels, self.input_resolution, self.input_resolution)
        
        # 2. U-Net编码器
        x1 = self.inc(x)      # [batch, features, 32, 32]
        x2 = self.down1(x1)   # [batch, features*2, 16, 16]
        x3 = self.down2(x2)   # [batch, features*4, 8, 8]
        x4 = self.down3(x3)   # [batch, features*8, 4, 4]
        x5 = self.down4(x4)   # [batch, features*16, 2, 2]
        
        # 3. U-Net解码器
        x = self.up1(x5, x4)  # [batch, features*8, 4, 4]
        x = self.up2(x, x3)   # [batch, features*4, 8, 8]
        x = self.up3(x, x2)   # [batch, features*2, 16, 16]
        x = self.up4(x, x1)   # [batch, features, 32, 32]
        
        # 4. 输出层
        x = self.outc(x)      # [batch, out_channels, 32, 32]
        
        # 5. 上采样到目标分辨率
        if x.shape[-1] != self.output_resolution:
            x = F.interpolate(
                x, 
                size=(self.output_resolution, self.output_resolution),
                mode='bilinear',
                align_corners=False
            )  # [batch, out_channels, 128, 128]
        
        # 6. 展平输出 [batch, output_dim]
        output = x.view(batch_size, -1)
        
        # 7. 确保输出维度正确
        if output.shape[1] != self.output_dim:
            # 使用线性插值调整到精确的输出维度
            output = F.interpolate(
                output.unsqueeze(1), 
                size=self.output_dim, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
        
        return output

class FixedEnhancedUNet2d(FixedEnhancedUNet):
    """2D版本的修复增强U-Net"""
    
    def __init__(self, 
                 in_channels=1,
                 out_channels=1,
                 init_features=64,
                 input_resolution=32,
                 output_resolution=128,
                 bilinear=False,
                 **kwargs):
        # 计算实际的输入输出维度
        input_dim = input_resolution * input_resolution * in_channels
        output_dim = output_resolution * output_resolution * out_channels
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            bilinear=bilinear,
            **kwargs
        )

# 工厂函数
def create_fixed_enhanced_unet2d(in_channels=1,
                                 out_channels=1,
                                 init_features=64,
                                 input_resolution=32,
                                 output_resolution=128,
                                 bilinear=False,
                                 **kwargs) -> FixedEnhancedUNet2d:
    """创建修复版增强U-Net2d模型"""
    return FixedEnhancedUNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        bilinear=bilinear,
        **kwargs
    )

def create_fixed_enhanced_unet(input_dim=1024,
                               output_dim=16384,
                               in_channels=1,
                               out_channels=1,
                               init_features=64,
                               bilinear=False,
                               **kwargs) -> FixedEnhancedUNet:
    """创建修复版增强U-Net模型"""
    return FixedEnhancedUNet(
        input_dim=input_dim,
        output_dim=output_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        bilinear=bilinear,
        **kwargs
    )

if __name__ == "__main__":
    # 测试修复版U-Net模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("测试修复版FixedEnhancedUNet...")
    model = create_fixed_enhanced_unet(
        input_dim=1024,
        output_dim=16384,
        in_channels=1,
        out_channels=1,
        init_features=64
    ).to(device)
    
    # 测试输入：[batch, 1024]
    x = torch.randn(4, 1024).to(device)
    output = model(x)
    print(f"输入形状: {x.shape}, 输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n测试修复版FixedEnhancedUNet2d...")
    model_2d = create_fixed_enhanced_unet2d(
        in_channels=1,
        out_channels=1,
        input_resolution=32,
        output_resolution=128,
        init_features=64
    ).to(device)
    
    # 测试输入：[batch, 1024]
    x_2d = torch.randn(4, 1024).to(device)
    output_2d = model_2d(x_2d)
    print(f"2D输入形状: {x_2d.shape}, 输出形状: {output_2d.shape}")
    print(f"2D参数数量: {sum(p.numel() for p in model_2d.parameters()):,}")
    
    print("\n所有测试通过！")