"""增强的Transformer模型，支持稀疏输入预测稠密输出功能"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Tuple, Optional, Union

# 添加mymodels路径
mymodels_path = Path(__file__).parent.parent / 'mymodels'
sys.path.insert(0, str(mymodels_path))

try:
    from transformer import TransformerFlowReconstructionModel
    HAS_TRANSFORMER = True
except ImportError as e:
    print(f"⚠️ Transformer模型导入失败: {e}")
    HAS_TRANSFORMER = False
    
    # 创建占位符类
    class TransformerFlowReconstructionModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.fc = nn.Linear(1, 1)
        
        def forward(self, x, time_steps=None):
            return torch.zeros(x.size(0), 1)

class EnhancedTransformer1d(nn.Module):
    """增强的Transformer1d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 input_channels=1,
                 output_channels=1,
                 d_model=128,
                 num_heads=4,
                 num_layers=3,
                 input_resolution=32,
                 output_resolution=128,
                 attention_type='simplified_self_attention',
                 pe_type='learnable_1d',
                 time_encoding='embedding',
                 max_time_steps=100,
                 use_upsampling=True):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_upsampling = use_upsampling
        
        # 计算输入输出维度
        self.input_dim = input_resolution * input_channels
        self.output_dim = output_resolution * output_channels
        
        # 创建底层Transformer模型
        self.transformer = TransformerFlowReconstructionModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_len=input_resolution,
            input_hw=None,  # 1D不需要hw
            pe_type=pe_type,
            output_head_type='global',
            time_encoding=time_encoding,
            max_time_steps=max_time_steps,
            attention_type=attention_type,
            use_memory_film=False,
            use_memory_concat=False
        )
        
        # 上采样层（如果需要）
        if self.use_upsampling and output_resolution > input_resolution:
            self.upsampler = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim * 2),
                nn.ReLU(),
                nn.Linear(self.output_dim * 2, output_resolution * output_channels)
            )
            self.output_dim = output_resolution * output_channels
    
    def forward(self, x):
        # x shape: [batch_size, input_resolution, input_channels]
        batch_size = x.size(0)
        
        # 展平输入
        x_flat = x.view(batch_size, -1)  # [batch_size, input_dim]
        
        # 创建时间步（简单设为0）
        time_steps = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
        
        # 通过Transformer
        output = self.transformer(x_flat, time_steps)  # [batch_size, output_dim]
        
        # 上采样（如果需要）
        if hasattr(self, 'upsampler'):
            output = self.upsampler(output)
        
        return output

class EnhancedTransformer2d(nn.Module):
    """增强的Transformer2d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 input_channels=1,
                 output_channels=1,
                 d_model=128,
                 num_heads=4,
                 num_layers=3,
                 input_resolution=(32, 32),
                 output_resolution=(128, 128),
                 attention_type='simplified_self_attention',
                 pe_type='learnable_2d',
                 time_encoding='embedding',
                 max_time_steps=100,
                 use_upsampling=True):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_upsampling = use_upsampling
        
        # 计算输入输出维度
        self.input_dim = input_resolution[0] * input_resolution[1] * input_channels
        self.output_dim = output_resolution[0] * output_resolution[1] * output_channels
        
        # 序列长度（用于位置编码）
        self.seq_len = input_resolution[0] * input_resolution[1]
        
        # 创建底层Transformer模型
        self.transformer = TransformerFlowReconstructionModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_len=self.seq_len,
            input_hw=input_resolution,
            pe_type=pe_type,
            output_head_type='global',
            time_encoding=time_encoding,
            max_time_steps=max_time_steps,
            attention_type=attention_type,
            use_memory_film=False,
            use_memory_concat=False
        )
        
        # 上采样层（如果需要）
        if self.use_upsampling and (output_resolution[0] > input_resolution[0] or 
                                   output_resolution[1] > input_resolution[1]):
            self.upsampler = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim * 2),
                nn.ReLU(),
                nn.Linear(self.output_dim * 2, output_resolution[0] * output_resolution[1] * output_channels)
            )
            self.output_dim = output_resolution[0] * output_resolution[1] * output_channels
    
    def forward(self, x):
        # x shape: [batch_size, H, W, input_channels]
        batch_size = x.size(0)
        
        # 展平输入
        x_flat = x.view(batch_size, -1)  # [batch_size, input_dim]
        
        # 创建时间步（简单设为0）
        time_steps = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
        
        # 通过Transformer
        output = self.transformer(x_flat, time_steps)  # [batch_size, output_dim]
        
        # 上采样（如果需要）
        if hasattr(self, 'upsampler'):
            output = self.upsampler(output)
        
        return output

# 工厂函数
def create_enhanced_transformer1d(input_channels=1,
                                 output_channels=1,
                                 d_model=128,
                                 num_heads=4,
                                 num_layers=3,
                                 input_resolution=32,
                                 output_resolution=128,
                                 attention_type='simplified_self_attention',
                                 pe_type='learnable_1d',
                                 time_encoding='embedding',
                                 max_time_steps=100,
                                 use_upsampling=True,
                                 **kwargs) -> EnhancedTransformer1d:
    """创建增强的Transformer1d模型"""
    return EnhancedTransformer1d(
        input_channels=input_channels,
        output_channels=output_channels,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        attention_type=attention_type,
        pe_type=pe_type,
        time_encoding=time_encoding,
        max_time_steps=max_time_steps,
        use_upsampling=use_upsampling
    )

def create_enhanced_transformer2d(input_channels=1,
                                 output_channels=1,
                                 d_model=128,
                                 num_heads=4,
                                 num_layers=3,
                                 input_resolution=(32, 32),
                                 output_resolution=(128, 128),
                                 attention_type='simplified_self_attention',
                                 pe_type='learnable_2d',
                                 time_encoding='embedding',
                                 max_time_steps=100,
                                 use_upsampling=True,
                                 **kwargs) -> EnhancedTransformer2d:
    """创建增强的Transformer2d模型"""
    return EnhancedTransformer2d(
        input_channels=input_channels,
        output_channels=output_channels,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        attention_type=attention_type,
        pe_type=pe_type,
        time_encoding=time_encoding,
        max_time_steps=max_time_steps,
        use_upsampling=use_upsampling
    )

if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if HAS_TRANSFORMER:
        # 测试Transformer1d
        print("测试EnhancedTransformer1d...")
        model_1d = create_enhanced_transformer1d(
            input_channels=1,
            input_resolution=32,
            output_resolution=128
        ).to(device)
        
        x_1d = torch.randn(4, 32, 1).to(device)
        output_1d = model_1d(x_1d)
        print(f"Transformer1d输入形状: {x_1d.shape}, 输出形状: {output_1d.shape}")
        
        # 测试Transformer2d
        print("\n测试EnhancedTransformer2d...")
        model_2d = create_enhanced_transformer2d(
            input_channels=1,
            input_resolution=(32, 32),
            output_resolution=(128, 128)
        ).to(device)
        
        x_2d = torch.randn(4, 32, 32, 1).to(device)
        output_2d = model_2d(x_2d)
        print(f"Transformer2d输入形状: {x_2d.shape}, 输出形状: {output_2d.shape}")
        
        print("\n所有测试通过！")
    else:
        print("⚠️ Transformer模型未正确导入，跳过测试")