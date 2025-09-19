"""模型优化工具，提供内存和计算效率优化功能"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Union, Tuple, Any
import warnings

class MemoryOptimizedModule(nn.Module):
    """内存优化模块基类"""
    
    def __init__(self, use_checkpoint: bool = False, use_amp: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp
        
    def forward_with_checkpoint(self, func, *args, **kwargs):
        """使用梯度检查点的前向传播"""
        if self.use_checkpoint and self.training:
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def enable_memory_optimization(self, checkpoint: bool = True, amp: bool = True):
        """启用内存优化"""
        self.use_checkpoint = checkpoint
        self.use_amp = amp
        
        if amp:
            # 将模型转换为半精度（除了某些层）
            self._convert_to_half_precision()
    
    def _convert_to_half_precision(self):
        """转换为半精度，但保持某些层为全精度"""
        for name, module in self.named_modules():
            # 保持BatchNorm和LayerNorm为全精度
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                 nn.LayerNorm, nn.GroupNorm)):
                continue
            # 保持损失函数相关层为全精度
            if 'loss' in name.lower() or 'criterion' in name.lower():
                continue
            
            # 其他层转换为半精度
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dtype == torch.float32:
                    module.half()

class OptimizedLinear(MemoryOptimizedModule):
    """优化的线性层"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 use_checkpoint: bool = False, use_amp: bool = False):
        super().__init__(use_checkpoint, use_amp)
        self.linear = nn.Linear(in_features, out_features, bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_checkpoint(self.linear, x)

class OptimizedConv1d(MemoryOptimizedModule):
    """优化的1D卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 use_checkpoint: bool = False, use_amp: bool = False):
        super().__init__(use_checkpoint, use_amp)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_checkpoint(self.conv, x)

class OptimizedConv2d(MemoryOptimizedModule):
    """优化的2D卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, 
                 padding: Union[int, Tuple[int, int]] = 0, bias: bool = True,
                 use_checkpoint: bool = False, use_amp: bool = False):
        super().__init__(use_checkpoint, use_amp)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_checkpoint(self.conv, x)

class OptimizedConv3d(MemoryOptimizedModule):
    """优化的3D卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1, 
                 padding: Union[int, Tuple[int, int, int]] = 0, bias: bool = True,
                 use_checkpoint: bool = False, use_amp: bool = False):
        super().__init__(use_checkpoint, use_amp)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_checkpoint(self.conv, x)

class MemoryEfficientAttention(MemoryOptimizedModule):
    """内存高效的注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 use_checkpoint: bool = False, use_amp: bool = False):
        super().__init__(use_checkpoint, use_amp)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = OptimizedLinear(embed_dim, embed_dim, use_checkpoint=use_checkpoint, use_amp=use_amp)
        self.k_proj = OptimizedLinear(embed_dim, embed_dim, use_checkpoint=use_checkpoint, use_amp=use_amp)
        self.v_proj = OptimizedLinear(embed_dim, embed_dim, use_checkpoint=use_checkpoint, use_amp=use_amp)
        self.out_proj = OptimizedLinear(embed_dim, embed_dim, use_checkpoint=use_checkpoint, use_amp=use_amp)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # 计算Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用Flash Attention风格的计算（如果可用）
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 的优化注意力
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # 传统注意力计算
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        return self.out_proj(attn_output)

class AdaptivePooling(nn.Module):
    """自适应池化层，用于处理不同分辨率的输入"""
    
    def __init__(self, output_size: Union[int, Tuple[int, ...]], mode: str = 'adaptive_avg'):
        super().__init__()
        self.output_size = output_size
        self.mode = mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'adaptive_avg':
            if len(x.shape) == 3:  # 1D
                return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size)
            elif len(x.shape) == 4:  # 2D
                return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)
            elif len(x.shape) == 5:  # 3D
                return torch.nn.functional.adaptive_avg_pool3d(x, self.output_size)
        elif self.mode == 'adaptive_max':
            if len(x.shape) == 3:  # 1D
                return torch.nn.functional.adaptive_max_pool1d(x, self.output_size)
            elif len(x.shape) == 4:  # 2D
                return torch.nn.functional.adaptive_max_pool2d(x, self.output_size)
            elif len(x.shape) == 5:  # 3D
                return torch.nn.functional.adaptive_max_pool3d(x, self.output_size)
        
        return x

class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0
        
    def __enter__(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            self.peak_memory = torch.cuda.max_memory_allocated(self.device)
    
    def get_peak_memory_mb(self) -> float:
        """获取峰值内存使用量(MB)"""
        return self.peak_memory / 1024 / 1024

def optimize_model_memory(model: nn.Module, enable_checkpoint: bool = True, 
                         enable_amp: bool = True) -> nn.Module:
    """优化模型内存使用"""
    
    def apply_optimization(module):
        if hasattr(module, 'enable_memory_optimization'):
            module.enable_memory_optimization(enable_checkpoint, enable_amp)
        
        for child in module.children():
            apply_optimization(child)
    
    apply_optimization(model)
    return model

def get_model_complexity(model: nn.Module, input_shape: Tuple[int, ...], 
                        device: torch.device) -> dict:
    """分析模型复杂度"""
    model.eval()
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小(MB)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    # 测量推理时间和内存
    dummy_input = torch.randn(*input_shape).to(device)
    
    with MemoryProfiler(device) as profiler:
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        # 测量时间
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                output = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / 10
    
    peak_memory = profiler.get_peak_memory_mb()
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': param_size,
        'avg_inference_time_s': avg_inference_time,
        'peak_memory_mb': peak_memory,
        'output_shape': tuple(output.shape),
        'flops_estimate': estimate_flops(model, dummy_input)
    }

def estimate_flops(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """估算模型的FLOPs"""
    flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal flops
        
        if isinstance(module, nn.Linear):
            # Linear layer: input_size * output_size * batch_size
            flops += input[0].numel() * module.out_features
        
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Convolution: output_elements * kernel_size * input_channels
            kernel_flops = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            if hasattr(module, 'kernel_size') and len(module.kernel_size) > 1:
                for k in module.kernel_size[1:]:
                    kernel_flops *= k
            
            output_elements = output.numel()
            flops += output_elements * kernel_flops * module.in_channels
        
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            # Normalization: 2 * input_elements (mean and variance)
            flops += 2 * input[0].numel()
        
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
            # Activation: 1 * input_elements
            flops += input[0].numel()
    
    # 注册钩子
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(flop_count_hook))
    
    # 前向传播
    with torch.no_grad():
        model(input_tensor)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return flops

def create_memory_efficient_model(model_class, *args, enable_optimization: bool = True, **kwargs):
    """创建内存高效的模型"""
    model = model_class(*args, **kwargs)
    
    if enable_optimization:
        model = optimize_model_memory(model)
    
    return model

# 导出的优化函数
__all__ = [
    'MemoryOptimizedModule',
    'OptimizedLinear', 
    'OptimizedConv1d',
    'OptimizedConv2d', 
    'OptimizedConv3d',
    'MemoryEfficientAttention',
    'AdaptivePooling',
    'MemoryProfiler',
    'optimize_model_memory',
    'get_model_complexity',
    'estimate_flops',
    'create_memory_efficient_model'
]