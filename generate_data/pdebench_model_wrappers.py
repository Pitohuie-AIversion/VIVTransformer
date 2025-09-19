
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDEBench兼容性包装器
处理不同模型的输入输出格式差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PDEBenchMLPWrapper(nn.Module):
    """MLP模型的PDEBench兼容性包装器"""
    
    def __init__(self, mlp_model):
        super().__init__()
        self.mlp_model = mlp_model
    
    def forward(self, x, time_steps=None):
        """
        Args:
            x: 输入数据 [batch, channels, h, w] 或 [batch, features]
            time_steps: 时间步信息（可选）
        Returns:
            输出数据 [batch, channels, h, w] 或 [batch, features]
        """
        original_shape = x.shape
        
        # 处理不同的输入格式
        if len(original_shape) == 4:  # [batch, channels, h, w]
            # 转换为MLP期望的格式 [batch, h, w, channels]
            x_mlp = x.permute(0, 2, 3, 1)
            output_mlp = self.mlp_model(x_mlp)
            
            # 转换回标准格式 [batch, channels, h, w]
            if len(output_mlp.shape) == 4:
                output = output_mlp.permute(0, 3, 1, 2)
            else:
                output = output_mlp
                
        elif len(original_shape) == 2:  # [batch, features]
            # 直接传递给MLP
            output = self.mlp_model(x)
            
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        return output

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
        # FNO模型直接处理标准格式
        return self.fno_model(x)

def create_compatible_mlp(input_channels=1, output_channels=1, 
                         input_resolution=(32, 32), output_resolution=(128, 128)):
    """创建兼容的MLP模型"""
    from models.enhanced_mlp import EnhancedMLP2d
    
    base_model = EnhancedMLP2d(
        input_channels=input_channels,
        output_channels=output_channels,
        input_resolution=input_resolution,
        output_resolution=output_resolution
    )
    
    return PDEBenchMLPWrapper(base_model)

def create_compatible_fno(num_channels=1, input_resolution=(32, 32), 
                         output_resolution=(128, 128)):
    """创建兼容的FNO模型"""
    from models.enhanced_fno import EnhancedFNO2d
    
    base_model = EnhancedFNO2d(
        num_channels=num_channels,
        input_resolution=input_resolution,
        output_resolution=output_resolution
    )
    
    return PDEBenchFNOWrapper(base_model)
