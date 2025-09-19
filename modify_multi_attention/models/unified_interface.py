"""统一模型接口规范"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Dict, Any
import warnings
from optimization_utils import get_model_complexity, MemoryProfiler

class BaseEnhancedModel(nn.Module, ABC):
    """增强模型基类，定义统一接口"""
    
    def __init__(self, 
                 input_resolution: Union[int, Tuple[int, ...]], 
                 output_resolution: Union[int, Tuple[int, ...]],
                 input_channels: int = 1,
                 output_channels: int = 1,
                 **kwargs):
        super().__init__()
        
        # 标准化分辨率格式
        self.input_resolution = self._normalize_resolution(input_resolution)
        self.output_resolution = self._normalize_resolution(output_resolution)
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 模型元信息
        self.model_type = self.__class__.__name__
        self.dimension = self._infer_dimension()
        
        # 性能统计
        self._complexity_cache = None
        self._last_input_shape = None
        
    def _normalize_resolution(self, resolution: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """标准化分辨率格式"""
        if isinstance(resolution, int):
            return (resolution,)
        return tuple(resolution)
    
    def _infer_dimension(self) -> int:
        """推断模型维度"""
        return len(self.input_resolution)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，子类必须实现"""
        pass
    
    def validate_input(self, x: torch.Tensor) -> bool:
        """验证输入张量的形状和类型"""
        expected_dims = 2 + self.dimension  # batch + channels + spatial dims
        
        if x.dim() != expected_dims:
            raise ValueError(
                f"Expected {expected_dims}D tensor (batch, channels, *spatial), "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )
        
        if x.size(1) != self.input_channels:
            warnings.warn(
                f"Input channels mismatch: expected {self.input_channels}, "
                f"got {x.size(1)}. Model may not work correctly."
            )
        
        # 检查空间维度
        spatial_dims = x.shape[2:]
        if len(spatial_dims) != len(self.input_resolution):
            raise ValueError(
                f"Spatial dimensions mismatch: expected {len(self.input_resolution)}D, "
                f"got {len(spatial_dims)}D"
            )
        
        return True
    
    def get_expected_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """根据输入形状计算期望的输出形状"""
        batch_size = input_shape[0]
        output_shape = [batch_size, self.output_channels] + list(self.output_resolution)
        return tuple(output_shape)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': self.model_type,
            'dimension': self.dimension,
            'input_resolution': self.input_resolution,
            'output_resolution': self.output_resolution,
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_complexity_analysis(self, input_shape: Optional[Tuple[int, ...]] = None, 
                              device: Optional[torch.device] = None) -> Dict[str, Any]:
        """获取模型复杂度分析"""
        if device is None:
            device = next(self.parameters()).device
        
        if input_shape is None:
            # 使用默认输入形状
            input_shape = (1, self.input_channels) + self.input_resolution
        
        # 检查缓存
        if (self._complexity_cache is not None and 
            self._last_input_shape == input_shape):
            return self._complexity_cache
        
        # 计算复杂度
        complexity = get_model_complexity(self, input_shape, device)
        
        # 缓存结果
        self._complexity_cache = complexity
        self._last_input_shape = input_shape
        
        return complexity
    
    def benchmark(self, input_shape: Optional[Tuple[int, ...]] = None,
                 num_runs: int = 100, warmup_runs: int = 10,
                 device: Optional[torch.device] = None) -> Dict[str, float]:
        """性能基准测试"""
        if device is None:
            device = next(self.parameters()).device
        
        if input_shape is None:
            input_shape = (1, self.input_channels) + self.input_resolution
        
        self.eval()
        dummy_input = torch.randn(*input_shape).to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self(dummy_input)
        
        # 同步GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 测量推理时间
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        # 测量内存使用
        with MemoryProfiler(device) as profiler:
            with torch.no_grad():
                _ = self(dummy_input)
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': 1.0 / avg_time,
            'peak_memory_mb': profiler.get_peak_memory_mb()
        }
    
    def summary(self, input_shape: Optional[Tuple[int, ...]] = None, 
               device: Optional[torch.device] = None) -> str:
        """生成模型摘要"""
        info = self.get_model_info()
        complexity = self.get_complexity_analysis(input_shape, device)
        
        summary_lines = [
            f"模型类型: {info['model_type']}",
            f"维度: {info['dimension']}D",
            f"输入分辨率: {info['input_resolution']}",
            f"输出分辨率: {info['output_resolution']}",
            f"输入通道: {info['input_channels']}",
            f"输出通道: {info['output_channels']}",
            f"总参数数: {info['total_parameters']:,}",
            f"可训练参数数: {info['trainable_parameters']:,}",
            f"模型大小: {complexity['model_size_mb']:.2f} MB",
            f"估计FLOPs: {complexity['flops_estimate']:,}",
            f"推理时间: {complexity['avg_inference_time_s']*1000:.2f} ms",
            f"峰值内存: {complexity['peak_memory_mb']:.2f} MB"
        ]
        
        return "\n".join(summary_lines)
    
    def save_model(self, filepath: str, include_optimizer: bool = False, 
                  optimizer: Optional[torch.optim.Optimizer] = None):
        """保存模型"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'model_class': self.__class__.__name__
        }
        
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[torch.device] = None):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        
        # 从模型信息重建模型
        model_info = checkpoint['model_info']
        model = cls(
            input_resolution=model_info['input_resolution'],
            output_resolution=model_info['output_resolution'],
            input_channels=model_info['input_channels'],
            output_channels=model_info['output_channels']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        return model

class ModelRegistry:
    """模型注册表，管理所有可用模型"""
    
    _models = {}
    
    @classmethod
    def register(cls, name: str, model_class: type, creator_func: callable):
        """注册模型"""
        cls._models[name] = {
            'class': model_class,
            'creator': creator_func
        }
    
    @classmethod
    def create_model(cls, name: str, **kwargs):
        """创建模型实例"""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available models: {list(cls._models.keys())}")
        
        return cls._models[name]['creator'](**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有可用模型"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, name: str) -> dict:
        """获取模型信息"""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        
        model_class = cls._models[name]['class']
        return {
            'name': name,
            'class_name': model_class.__name__,
            'module': model_class.__module__
        }

class UnifiedModelInterface:
    """统一模型接口，提供一致的API"""
    
    def __init__(self, model: BaseEnhancedModel):
        self.model = model
    
    def predict(self, x: torch.Tensor, 
               validate_input: bool = True,
               return_numpy: bool = False) -> Union[torch.Tensor, 'numpy.ndarray']:
        """统一预测接口"""
        if validate_input:
            self.model.validate_input(x)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        if return_numpy:
            return output.cpu().numpy()
        
        return output
    
    def fit(self, train_loader, val_loader=None, epochs: int = 100,
           optimizer=None, criterion=None, device=None, **kwargs):
        """统一训练接口"""
        if device is None:
            device = next(self.model.parameters()).device
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion, device)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")
    
    def evaluate(self, data_loader, criterion=None, device=None) -> float:
        """统一评估接口"""
        if device is None:
            device = next(self.model.parameters()).device
        
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def get_model_summary(self) -> str:
        """获取模型摘要"""
        return self.model.summary()
    
    def benchmark_model(self, **kwargs) -> Dict[str, float]:
        """模型性能基准测试"""
        return self.model.benchmark(**kwargs)

# 工具函数
def create_unified_interface(model_name: str, **kwargs) -> UnifiedModelInterface:
    """创建统一接口的模型"""
    model = ModelRegistry.create_model(model_name, **kwargs)
    return UnifiedModelInterface(model)

def compare_models(models: list, input_shape: Tuple[int, ...], 
                  device: Optional[torch.device] = None) -> Dict[str, Dict[str, Any]]:
    """比较多个模型的性能"""
    results = {}
    
    for model in models:
        if isinstance(model, str):
            # 如果是字符串，从注册表创建模型
            model_instance = ModelRegistry.create_model(model)
        else:
            model_instance = model
        
        if device is not None:
            model_instance = model_instance.to(device)
        
        # 获取模型信息和复杂度分析
        info = model_instance.get_model_info()
        complexity = model_instance.get_complexity_analysis(input_shape, device)
        benchmark = model_instance.benchmark(input_shape, device=device)
        
        results[info['model_type']] = {
            **info,
            **complexity,
            **benchmark
        }
    
    return results

# 导出的接口
__all__ = [
    'BaseEnhancedModel',
    'ModelRegistry', 
    'UnifiedModelInterface',
    'create_unified_interface',
    'compare_models'
]