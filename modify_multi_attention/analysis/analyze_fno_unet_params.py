#!/usr/bin/env python3
"""
FNO和UNet模型参数量详细分析脚本
分析各层参数量构成，找到最小配置
"""

import torch
import torch.nn as nn
import sys
import os

# 添加模型路径
sys.path.append('models')
from enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
from enhanced_unet import EnhancedUNet1d, EnhancedUNet2d

def count_parameters_by_layer(model, model_name):
    """按层统计参数量"""
    print(f"\n=== {model_name} 参数量详细分析 ===")
    total_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:40s}: {param_count:>10,} 参数")
    
    print(f"{'总参数量':40s}: {total_params:>10,} 参数")
    return total_params

def test_minimal_fno_configs():
    """测试FNO的最小配置"""
    print("🔍 测试FNO最小配置...")
    
    configs = [
        {"width": 2, "modes": 1, "num_layers": 1},
        {"width": 4, "modes": 2, "num_layers": 1},
        {"width": 8, "modes": 2, "num_layers": 1},
        {"width": 16, "modes": 4, "num_layers": 1},
    ]
    
    for config in configs:
        try:
            # 创建简化的FNO模型
            model = SimpleFNO(
                input_dim=1024,
                output_dim=16384,
                width=config["width"],
                modes=config["modes"]
            )
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"FNO配置 {config}: {param_count:,} 参数")
            
            # 测试前向传播
            x = torch.randn(2, 1024)
            y = model(x)
            print(f"  输入: {x.shape}, 输出: {y.shape}")
            
        except Exception as e:
            print(f"FNO配置 {config} 失败: {e}")

def test_minimal_unet_configs():
    """测试UNet的最小配置"""
    print("\n🔍 测试UNet最小配置...")
    
    configs = [
        {"base_channels": 2, "num_levels": 1},
        {"base_channels": 4, "num_levels": 1},
        {"base_channels": 8, "num_levels": 2},
        {"base_channels": 16, "num_levels": 2},
    ]
    
    for config in configs:
        try:
            # 创建简化的UNet模型
            model = SimpleUNet(
                input_dim=1024,
                output_dim=16384,
                base_channels=config["base_channels"],
                num_levels=config["num_levels"]
            )
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"UNet配置 {config}: {param_count:,} 参数")
            
            # 测试前向传播
            x = torch.randn(2, 1024)
            y = model(x)
            print(f"  输入: {x.shape}, 输出: {y.shape}")
            
        except Exception as e:
            print(f"UNet配置 {config} 失败: {e}")

class SimpleFNO(nn.Module):
    """极简FNO模型，只保留核心功能"""
    
    def __init__(self, input_dim, output_dim, width=4, modes=2):
        super().__init__()
        
        self.width = width
        self.modes = modes
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, width)
        
        # 简化的频域卷积层
        self.spectral_conv = nn.Conv1d(width, width, 1)
        self.regular_conv = nn.Conv1d(width, width, 1)
        
        # 输出投影
        self.output_proj = nn.Linear(width, output_dim)
        
    def forward(self, x):
        # x: [batch, input_dim]
        batch_size = x.shape[0]
        
        # 输入投影
        x = self.input_proj(x)  # [batch, width]
        x = x.unsqueeze(-1)  # [batch, width, 1]
        
        # 简化的频域处理
        x_conv = self.regular_conv(x)
        x = x + x_conv
        x = torch.relu(x)
        
        # 输出投影
        x = x.squeeze(-1)  # [batch, width]
        x = self.output_proj(x)  # [batch, output_dim]
        
        return x

class SimpleUNet(nn.Module):
    """极简UNet模型，只保留核心功能"""
    
    def __init__(self, input_dim, output_dim, base_channels=4, num_levels=1):
        super().__init__()
        
        self.base_channels = base_channels
        self.num_levels = num_levels
        
        # 输入投影到2D
        self.input_proj = nn.Linear(input_dim, base_channels * 32 * 32)
        
        # 编码器
        self.encoder = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_levels):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            in_ch = out_ch
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(num_levels):
            out_ch = base_channels * (2 ** (num_levels - i - 1))
            self.decoder.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            in_ch = out_ch
        
        # 输出投影
        self.output_proj = nn.Linear(base_channels * 32 * 32, output_dim)
        
    def forward(self, x):
        # x: [batch, input_dim]
        batch_size = x.shape[0]
        
        # 投影到2D
        x = self.input_proj(x)  # [batch, base_channels * 32 * 32]
        x = x.view(batch_size, self.base_channels, 32, 32)  # [batch, base_channels, 32, 32]
        
        # 编码
        for conv in self.encoder:
            x = torch.relu(conv(x))
        
        # 解码
        for conv in self.decoder:
            x = torch.relu(conv(x))
        
        # 输出投影
        x = x.view(batch_size, -1)  # [batch, base_channels * 32 * 32]
        x = self.output_proj(x)  # [batch, output_dim]
        
        return x

def main():
    """主函数"""
    print("🚀 开始FNO和UNet参数量详细分析...")
    
    # 分析现有模型
    print("\n📊 分析现有EnhancedFNO和EnhancedUNet模型...")
    
    try:
        # FNO1d分析
        fno1d = EnhancedFNO1d(
            num_channels=1,
            modes=4,
            width=16,
            input_resolution=32,
            output_resolution=128
        )
        count_parameters_by_layer(fno1d, "EnhancedFNO1d (width=16, modes=4)")
        
        # UNet1d分析
        unet1d = EnhancedUNet1d(
            in_channels=1,
            out_channels=1,
            init_features=8,
            input_resolution=32,
            output_resolution=128
        )
        count_parameters_by_layer(unet1d, "EnhancedUNet1d (init_features=8)")
        
    except Exception as e:
        print(f"现有模型分析失败: {e}")
    
    # 测试最小配置
    test_minimal_fno_configs()
    test_minimal_unet_configs()
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main()