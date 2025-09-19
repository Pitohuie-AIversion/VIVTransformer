"""PDEBench模型适配器，支持稀疏输入预测稠密输出功能"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import sys

# 添加路径以导入PDEBench模型
project_root = Path(__file__).parent.parent.parent
pdebench_path = project_root / 'PDEBench' / 'pdebench'
sys.path.insert(0, str(pdebench_path))

# 导入PDEBench模型
from models.fno.fno import FNO1d, FNO2d
from models.unet.unet import UNet1d, UNet2d

# 导入dynamic_resolution_trainer的数据集
sys.path.insert(0, str(project_root / 'generate_data'))
from dynamic_resolution_trainer import DynamicResolutionDataset

class SparseToDeseAdapter(nn.Module):
    """稀疏输入到稠密输出的适配器基类"""
    
    def __init__(self, 
                 input_resolution: Tuple[int, int],
                 output_resolution: Tuple[int, int],
                 interpolation_mode: str = 'bilinear'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.interpolation_mode = interpolation_mode
        
    def upsample_output(self, x: torch.Tensor) -> torch.Tensor:
        """将模型输出上采样到目标分辨率"""
        if len(x.shape) == 4:  # [B, C, H, W]
            return torch.nn.functional.interpolate(
                x, size=self.output_resolution, 
                mode=self.interpolation_mode, align_corners=False
            )
        elif len(x.shape) == 3:  # [B, C, L] for 1D
            return torch.nn.functional.interpolate(
                x, size=(self.output_resolution[0],), 
                mode='linear', align_corners=False
            )
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
            
    def downsample_input(self, x: torch.Tensor) -> torch.Tensor:
        """将输入下采样到模型所需分辨率"""
        if len(x.shape) == 4:  # [B, C, H, W]
            return torch.nn.functional.interpolate(
                x, size=self.input_resolution, 
                mode=self.interpolation_mode, align_corners=False
            )
        elif len(x.shape) == 3:  # [B, C, L] for 1D
            return torch.nn.functional.interpolate(
                x, size=(self.input_resolution[0],), 
                mode='linear', align_corners=False
            )
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

class AdaptedFNO2d(SparseToDeseAdapter):
    """适配的FNO2d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 num_channels: int = 1,
                 modes1: int = 12,
                 modes2: int = 12, 
                 width: int = 20,
                 initial_step: int = 10,
                 input_resolution: Tuple[int, int] = (64, 64),
                 output_resolution: Tuple[int, int] = (128, 128)):
        super().__init__(input_resolution, output_resolution)
        
        self.fno = FNO2d(
            num_channels=num_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            initial_step=initial_step
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入分辨率不匹配，先下采样
        if x.shape[-2:] != self.input_resolution:
            x = self.downsample_input(x)
            
        # FNO前向传播
        x = self.fno(x)
        
        # 上采样到目标分辨率
        x = self.upsample_output(x)
        
        return x

class AdaptedUNet2d(SparseToDeseAdapter):
    """适配的UNet2d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 1,
                 init_features: int = 32,
                 input_resolution: Tuple[int, int] = (64, 64),
                 output_resolution: Tuple[int, int] = (128, 128)):
        super().__init__(input_resolution, output_resolution)
        
        self.unet = UNet2d(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入分辨率不匹配，先下采样
        if x.shape[-2:] != self.input_resolution:
            x = self.downsample_input(x)
            
        # UNet前向传播
        x = self.unet(x)
        
        # 上采样到目标分辨率
        x = self.upsample_output(x)
        
        return x

class AdaptedFNO1d(SparseToDeseAdapter):
    """适配的FNO1d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 num_channels: int = 1,
                 modes: int = 16,
                 width: int = 64,
                 initial_step: int = 10,
                 input_resolution: Tuple[int, int] = (64, 1),
                 output_resolution: Tuple[int, int] = (128, 1)):
        super().__init__(input_resolution, output_resolution)
        
        self.fno = FNO1d(
            num_channels=num_channels,
            modes=modes,
            width=width,
            initial_step=initial_step
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入分辨率不匹配，先下采样
        if x.shape[-1] != self.input_resolution[0]:
            x = self.downsample_input(x)
            
        # FNO前向传播
        x = self.fno(x)
        
        # 上采样到目标分辨率
        x = self.upsample_output(x)
        
        return x

class AdaptedUNet1d(SparseToDeseAdapter):
    """适配的UNet1d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 1,
                 init_features: int = 32,
                 input_resolution: Tuple[int, int] = (64, 1),
                 output_resolution: Tuple[int, int] = (128, 1)):
        super().__init__(input_resolution, output_resolution)
        
        self.unet = UNet1d(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入分辨率不匹配，先下采样
        if x.shape[-1] != self.input_resolution[0]:
            x = self.downsample_input(x)
            
        # UNet前向传播
        x = self.unet(x)
        
        # 上采样到目标分辨率
        x = self.upsample_output(x)
        
        return x

class PDEBenchDatasetAdapter:
    """PDEBench数据集适配器，兼容dynamic_resolution_trainer的数据格式"""
    
    def __init__(self, 
                 data_path: str,
                 input_resolution: Tuple[int, int],
                 output_resolution: Tuple[int, int],
                 num_samples: int = 100,
                 normalize_data: bool = True):
        
        self.dataset = DynamicResolutionDataset(
            data_path=data_path,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            num_samples=num_samples,
            normalize_data=normalize_data
        )
        
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
        """获取数据加载器"""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
    def get_data_statistics(self):
        """获取数据统计信息"""
        return self.dataset.get_data_statistics()
        
    def __len__(self):
        return len(self.dataset)

def create_adapted_model(model_type: str, 
                        model_config: Dict[str, Any],
                        input_resolution: Tuple[int, int],
                        output_resolution: Tuple[int, int]) -> nn.Module:
    """创建适配的模型"""
    
    if model_type.lower() == 'fno2d':
        return AdaptedFNO2d(
            num_channels=model_config.get('num_channels', 1),
            modes1=model_config.get('modes1', 12),
            modes2=model_config.get('modes2', 12),
            width=model_config.get('width', 20),
            initial_step=model_config.get('initial_step', 10),
            input_resolution=input_resolution,
            output_resolution=output_resolution
        )
    elif model_type.lower() == 'unet2d':
        return AdaptedUNet2d(
            in_channels=model_config.get('in_channels', 3),
            out_channels=model_config.get('out_channels', 1),
            init_features=model_config.get('init_features', 32),
            input_resolution=input_resolution,
            output_resolution=output_resolution
        )
    elif model_type.lower() == 'fno1d':
        return AdaptedFNO1d(
            num_channels=model_config.get('num_channels', 1),
            modes=model_config.get('modes', 16),
            width=model_config.get('width', 64),
            initial_step=model_config.get('initial_step', 10),
            input_resolution=input_resolution,
            output_resolution=output_resolution
        )
    elif model_type.lower() == 'unet1d':
        return AdaptedUNet1d(
            in_channels=model_config.get('in_channels', 3),
            out_channels=model_config.get('out_channels', 1),
            init_features=model_config.get('init_features', 32),
            input_resolution=input_resolution,
            output_resolution=output_resolution
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# 示例使用
if __name__ == "__main__":
    # 创建适配的FNO2d模型
    model_config = {
        'num_channels': 1,
        'modes1': 12,
        'modes2': 12,
        'width': 20,
        'initial_step': 10
    }
    
    model = create_adapted_model(
        model_type='fno2d',
        model_config=model_config,
        input_resolution=(32, 32),
        output_resolution=(128, 128)
    )
    
    # 测试前向传播
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 创建数据集适配器
    # dataset_adapter = PDEBenchDatasetAdapter(
    #     data_path="path/to/data",
    #     input_resolution=(32, 32),
    #     output_resolution=(128, 128)
    # )
    # dataloader = dataset_adapter.get_dataloader(batch_size=16)