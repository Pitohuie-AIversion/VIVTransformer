#!/usr/bin/env python3
"""统一网络架构对比测试脚本"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
import time
import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import traceback

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'mymodels'))

try:
    from enhanced_transformer import create_enhanced_transformer1d, create_enhanced_transformer2d
    from mymodels.transformer import TransformerFlowReconstructionModel
except ImportError as e:
    print(f"导入错误: {e}")
    # 继续运行，但标记transformer不可用
    TRANSFORMER_AVAILABLE = False
else:
    TRANSFORMER_AVAILABLE = True

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果数据类"""
    model_name: str
    model_type: str
    dataset_name: str
    success: bool = False
    error: Optional[str] = None
    
    # 性能指标
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    relative_error: float = 0.0
    r2_score: float = 0.0
    
    # 时间和资源
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage_mb: float = 0.0
    param_count: int = 0
    
    # 形状信息
    input_shape: List[int] = None
    output_shape: List[int] = None
    expected_output_shape: List[int] = None
    shape_match: bool = False

class SyntheticDataGenerator:
    """合成数据生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('device', {}).get('use_cuda', True) else 'cpu')
    
    def generate_time_series(self, dataset_config: Dict[str, Any], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成时间序列数据"""
        length = dataset_config['length']
        noise_level = dataset_config['noise_level']
        
        # 生成输入数据（正弦波 + 噪声）
        t = torch.linspace(0, 4*np.pi, length)
        x = torch.sin(t) + torch.sin(3*t) * 0.5
        x = x + torch.randn_like(x) * noise_level
        
        # 生成目标数据（相位偏移）
        y = torch.sin(t + np.pi/4) + torch.sin(3*t + np.pi/4) * 0.5
        
        # 扩展到批次
        x = x.unsqueeze(0).repeat(batch_size, 1)
        y = y.unsqueeze(0).repeat(batch_size, 1)
        
        return x.to(self.device), y.to(self.device)
    
    def generate_image_data(self, dataset_config: Dict[str, Any], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成图像数据"""
        size = dataset_config['size']
        noise_level = dataset_config['noise_level']
        
        # 生成输入图像（高斯分布）
        x = torch.randn(batch_size, 1, size[0], size[1])
        
        # 生成目标图像（简单变换）
        y = torch.roll(x, shifts=1, dims=-1)  # 水平移动
        y = y + torch.randn_like(y) * noise_level
        
        return x.to(self.device), y.to(self.device)
    
    def generate_pde_data(self, dataset_config: Dict[str, Any], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成PDE数据"""
        # 简化的扩散方程数据
        domain = dataset_config['domain']
        time_range = dataset_config['time_range']
        
        # 空间网格
        x = torch.linspace(domain[0], domain[1], 64)
        t = torch.linspace(time_range[0], time_range[1], 10)
        
        # 初始条件（高斯分布）
        x0 = torch.exp(-((x - 0.5) ** 2) / 0.1)
        
        # 简化的解析解（扩散）
        X, T = torch.meshgrid(x, t, indexing='ij')
        solution = torch.exp(-((X - 0.5) ** 2) / (0.1 + T * 0.1))
        
        # 扩展到批次
        input_data = x0.unsqueeze(0).repeat(batch_size, 1)
        target_data = solution[:, -1].unsqueeze(0).repeat(batch_size, 1)
        
        return input_data.to(self.device), target_data.to(self.device)
    
    def generate_time_series_data(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成时间序列数据（简化接口）"""
        dataset_config = {
            'length': seq_len,
            'noise_level': 0.1
        }
        return self.generate_time_series(dataset_config, batch_size)

class ModelFactory:
    """模型工厂类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('device', {}).get('use_cuda', True) else 'cpu')
    
    def create_mlp(self, model_config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
        """创建MLP模型"""
        hidden_layers = model_config['hidden_layers']
        activation = model_config.get('activation', 'relu')
        dropout = model_config.get('dropout', 0.0)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def create_simple_unet1d(self, model_config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
        """创建简化的1D UNet模型"""
        features = model_config['features']
        
        class SimpleUNet1D(nn.Module):
            def __init__(self, in_channels, out_channels, features):
                super().__init__()
                self.encoder = nn.ModuleList()
                self.decoder = nn.ModuleList()
                
                # 编码器
                prev_feat = in_channels
                for feat in features:
                    self.encoder.append(nn.Sequential(
                        nn.Conv1d(prev_feat, feat, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(feat, feat, 3, padding=1),
                        nn.ReLU()
                    ))
                    prev_feat = feat
                
                # 解码器
                for i in range(len(features)-1, 0, -1):
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose1d(features[i], features[i-1], 2, stride=2),
                        nn.ReLU()
                    ))
                
                self.final_conv = nn.Conv1d(features[0], out_channels, 1)
            
            def forward(self, x):
                # 重塑输入
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)  # [B, 1, L]
                
                # 编码
                skip_connections = []
                for encoder in self.encoder:
                    x = encoder(x)
                    skip_connections.append(x)
                    x = nn.functional.max_pool1d(x, 2)
                
                # 解码
                for i, decoder in enumerate(self.decoder):
                    x = decoder(x)
                    if i < len(skip_connections) - 1:
                        skip = skip_connections[-(i+2)]
                        if x.shape[-1] != skip.shape[-1]:
                            x = nn.functional.interpolate(x, size=skip.shape[-1])
                        x = x + skip
                
                x = self.final_conv(x)
                return x.squeeze(1)  # [B, L]
        
        return SimpleUNet1D(1, 1, features)
    
    def create_simple_fno1d(self, model_config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
        """创建简化的1D FNO模型"""
        modes = model_config['modes']
        width = model_config['width']
        layers = model_config['layers']
        
        class SimpleFNO1D(nn.Module):
            def __init__(self, modes, width, layers, input_dim, output_dim):
                super().__init__()
                self.modes = modes
                self.width = width
                self.layers = layers
                
                self.fc0 = nn.Linear(1, width)
                
                self.conv_layers = nn.ModuleList()
                for _ in range(layers):
                    self.conv_layers.append(nn.Conv1d(width, width, 3, padding=1))
                
                self.fc1 = nn.Linear(width, 128)
                self.fc2 = nn.Linear(128, output_dim)
            
            def forward(self, x):
                if len(x.shape) == 2:
                    x = x.unsqueeze(-1)  # [B, L, 1]
                
                x = self.fc0(x)  # [B, L, width]
                x = x.permute(0, 2, 1)  # [B, width, L]
                
                for conv in self.conv_layers:
                    x = torch.relu(conv(x))
                
                x = x.permute(0, 2, 1)  # [B, L, width]
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                
                if len(x.shape) == 3 and x.shape[-1] == 1:
                    x = x.squeeze(-1)
                
                return x
        
        return SimpleFNO1D(modes, width, layers, input_dim, output_dim)
    
    def create_transformer(self, model_config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
        """创建Transformer模型"""
        if not TRANSFORMER_AVAILABLE:
            raise ImportError("Transformer模块不可用")
        
        return TransformerFlowReconstructionModel(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            attention_type=model_config.get('attention_type', 'simplified_self'),
            pe_type=model_config.get('pe_type', 'learnable_1d'),
            time_encoding=model_config.get('time_encoding', 'embedding'),
            max_time_steps=model_config.get('max_time_steps', 100),
            seq_len=input_dim,
            input_hw=None,
            output_head_type='global',
            use_memory_film=False,
            use_memory_concat=False
        )
    
    def create_model(self, model_name: str, model_config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
        """根据配置创建模型"""
        model_type = model_config.get('model_type', '')
        
        try:
            # 导入增强模型
            from enhanced_fno import create_enhanced_fno1d, create_enhanced_fno2d
            from enhanced_unet import create_enhanced_unet1d, create_enhanced_unet2d
            from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d
            from enhanced_pinn import create_enhanced_pinn1d, create_enhanced_pinn2d
            from enhanced_transformer import create_enhanced_transformer1d, create_enhanced_transformer2d
            
            if model_type == 'fno':
                # 根据输入维度选择1D或2D FNO
                input_resolution = model_config.get('input_resolution', 64)
                output_resolution = model_config.get('output_resolution', 128)
                
                if isinstance(input_resolution, list) and len(input_resolution) == 2:
                    # 2D FNO
                    return create_enhanced_fno2d(
                        num_channels=model_config.get('num_channels', 1),
                        modes1=model_config.get('modes1', 12),
                        modes2=model_config.get('modes2', 12),
                        width=model_config.get('width', 20),
                        input_resolution=tuple(input_resolution),
                        output_resolution=tuple(model_config.get('output_resolution', [128, 128])),
                        use_upsampling=model_config.get('use_upsampling', True)
                    )
                else:
                    # 1D FNO
                    return create_enhanced_fno1d(
                        num_channels=model_config.get('num_channels', 1),
                        modes=model_config.get('modes', 16),
                        width=model_config.get('width', 64),
                        input_resolution=input_resolution,
                        output_resolution=output_resolution,
                        use_upsampling=model_config.get('use_upsampling', True)
                    )
                    
            elif model_type == 'unet':
                # UNet模型
                input_resolution = model_config.get('input_resolution', 64)
                output_resolution = model_config.get('output_resolution', 128)
                
                if isinstance(input_resolution, list) and len(input_resolution) == 2:
                    # 2D UNet
                    return create_enhanced_unet2d(
                        in_channels=model_config.get('in_channels', 1),
                        out_channels=model_config.get('out_channels', 1),
                        init_features=model_config.get('init_features', 32),
                        input_resolution=tuple(input_resolution),
                        output_resolution=tuple(model_config.get('output_resolution', [128, 128])),
                        use_upsampling=model_config.get('use_upsampling', True)
                    )
                else:
                    # 1D UNet
                    return create_enhanced_unet1d(
                        in_channels=model_config.get('in_channels', 1),
                        out_channels=model_config.get('out_channels', 1),
                        init_features=model_config.get('init_features', 32),
                        input_resolution=input_resolution,
                        output_resolution=output_resolution,
                        use_upsampling=model_config.get('use_upsampling', True)
                    )
                    
            elif model_type == 'mlp':
                # MLP模型
                input_resolution = model_config.get('input_resolution', 64)
                output_resolution = model_config.get('output_resolution', 128)
                
                if isinstance(input_resolution, list) and len(input_resolution) == 2:
                    # 2D MLP
                    return create_enhanced_mlp2d(
                        input_channels=model_config.get('input_channels', 1),
                        output_channels=model_config.get('output_channels', 1),
                        hidden_dim=model_config.get('hidden_dim', 256),
                        num_layers=model_config.get('num_layers', 8),
                        input_resolution=tuple(input_resolution),
                        output_resolution=tuple(model_config.get('output_resolution', [128, 128]))
                    )
                else:
                    # 1D MLP
                    return create_enhanced_mlp1d(
                        input_channels=model_config.get('input_channels', 1),
                        output_channels=model_config.get('output_channels', 1),
                        hidden_dim=model_config.get('hidden_dim', 256),
                        num_layers=model_config.get('num_layers', 8),
                        input_resolution=input_resolution,
                        output_resolution=output_resolution
                    )
                    
            elif model_type == 'pinn':
                # PINN模型
                input_resolution = model_config.get('input_resolution', 64)
                output_resolution = model_config.get('output_resolution', 128)
                
                if isinstance(input_resolution, list) and len(input_resolution) == 2:
                    # 2D PINN
                    return create_enhanced_pinn2d(
                        output_dim=model_config.get('output_dim', 1),
                        hidden_dim=model_config.get('hidden_dim', 256),
                        num_layers=model_config.get('num_layers', 8),
                        input_resolution=tuple(input_resolution),
                        output_resolution=tuple(model_config.get('output_resolution', [128, 128]))
                    )
                else:
                    # 1D PINN
                    return create_enhanced_pinn1d(
                        output_dim=model_config.get('output_dim', 1),
                        hidden_dim=model_config.get('hidden_dim', 256),
                        num_layers=model_config.get('num_layers', 8),
                        input_resolution=input_resolution,
                        output_resolution=output_resolution
                    )
                    
            elif model_type == 'transformer':
                # Transformer模型
                input_resolution = model_config.get('input_resolution', 64)
                output_resolution = model_config.get('output_resolution', 128)
                
                if isinstance(input_resolution, list) and len(input_resolution) == 2:
                    # 2D Transformer
                    return create_enhanced_transformer2d(
                        input_channels=model_config.get('input_channels', 1),
                        output_channels=model_config.get('output_channels', 1),
                        d_model=model_config.get('d_model', 128),
                        num_heads=model_config.get('num_heads', 4),
                        num_layers=model_config.get('num_layers', 3),
                        input_resolution=tuple(input_resolution),
                        output_resolution=tuple(model_config.get('output_resolution', [128, 128])),
                        attention_type=model_config.get('attention_type', 'simplified_self_attention'),
                        pe_type=model_config.get('pe_type', 'learnable_2d'),
                        time_encoding=model_config.get('time_encoding', 'embedding'),
                        max_time_steps=model_config.get('max_time_steps', 100),
                        use_upsampling=model_config.get('use_upsampling', True)
                    )
                else:
                    # 1D Transformer
                    return create_enhanced_transformer1d(
                        input_channels=model_config.get('input_channels', 1),
                        output_channels=model_config.get('output_channels', 1),
                        d_model=model_config.get('d_model', 128),
                        num_heads=model_config.get('num_heads', 4),
                        num_layers=model_config.get('num_layers', 3),
                        input_resolution=input_resolution,
                        output_resolution=output_resolution,
                        attention_type=model_config.get('attention_type', 'simplified_self_attention'),
                        pe_type=model_config.get('pe_type', 'learnable_1d'),
                        time_encoding=model_config.get('time_encoding', 'embedding'),
                        max_time_steps=model_config.get('max_time_steps', 100),
                        use_upsampling=model_config.get('use_upsampling', True)
                    )
            else:
                # 回退到简单模型实现
                if model_type == 'mlp':
                    return self.create_mlp(model_config, input_dim, output_dim)
                elif model_type == 'unet':
                    return self.create_simple_unet1d(model_config, input_dim, output_dim)
                elif model_type == 'fno':
                    return self.create_simple_fno1d(model_config, input_dim, output_dim)
                elif model_type == 'transformer':
                    return self.create_transformer(model_config, input_dim, output_dim)
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")
                    
        except ImportError as e:
            logger.warning(f"无法导入增强模型 {model_type}: {e}，使用简单实现")
            # 回退到简单模型实现
            if model_type == 'mlp':
                return self.create_mlp(model_config, input_dim, output_dim)
            elif model_type == 'unet':
                return self.create_simple_unet1d(model_config, input_dim, output_dim)
            elif model_type == 'fno':
                return self.create_simple_fno1d(model_config, input_dim, output_dim)
            elif model_type == 'transformer':
                return self.create_transformer(model_config, input_dim, output_dim)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

class UnifiedComparisonTester:
    """统一对比测试器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('device', {}).get('use_cuda', True) else 'cpu')
        
        self.data_generator = SyntheticDataGenerator(self.config)
        self.model_factory = ModelFactory(self.config)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"Transformer可用: {TRANSFORMER_AVAILABLE}")
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def count_parameters(self, model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算评估指标"""
        with torch.no_grad():
            mse = nn.functional.mse_loss(predictions, targets).item()
            mae = nn.functional.l1_loss(predictions, targets).item()
            rmse = np.sqrt(mse)
            
            # 相对误差
            relative_error = (torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)).mean().item()
            
            # R²分数
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'relative_error': relative_error,
                'r2_score': r2
            }
    
    def generate_dataset(self, dataset_config: Dict[str, Any], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成数据集"""
        dataset_type = dataset_config['type']
        
        if dataset_type == 'synthetic':
            if 'length' in dataset_config:  # 时间序列
                return self.data_generator.generate_time_series(dataset_config, batch_size)
            elif 'size' in dataset_config:  # 图像
                return self.data_generator.generate_image_data(dataset_config, batch_size)
        elif dataset_type == 'pde':
            return self.data_generator.generate_pde_data(dataset_config, batch_size)
        
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    def train_model(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor], 
                   val_data: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """训练模型"""
        train_config = self.config['training']
        optimizer_config = self.config['optimizer']
        loss_config = self.config['loss']
        
        # 设置优化器
        learning_rate = float(train_config['learning_rate'])
        weight_decay = float(train_config['weight_decay'])
        
        if optimizer_config['type'] == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', [0.9, 0.999])
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # 设置损失函数
        if loss_config['type'] == 'MSELoss':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        
        # 学习率调度器
        scheduler_config = train_config['scheduler']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(scheduler_config['factor']),
            patience=int(scheduler_config['patience']),
            min_lr=float(scheduler_config['min_lr'])
        )
        
        model.train()
        train_x, train_y = train_data
        val_x, val_y = val_data
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(train_config['epochs']):
            # 训练
            optimizer.zero_grad()
            
            if hasattr(model, 'forward') and 'time_steps' in str(model.forward.__code__.co_varnames):
                # Transformer模型需要时间步
                time_steps = torch.zeros(train_x.shape[0], 1, dtype=torch.long).to(self.device)
                outputs = model(train_x, time_steps)
            else:
                outputs = model(train_x)
            
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'forward') and 'time_steps' in str(model.forward.__code__.co_varnames):
                    time_steps = torch.zeros(val_x.shape[0], 1, dtype=torch.long).to(self.device)
                    val_outputs = model(val_x, time_steps)
                else:
                    val_outputs = model(val_x)
                val_loss = criterion(val_outputs, val_y).item()
            
            model.train()
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            min_delta = float(train_config['min_delta'])  # 确保是浮点数
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                patience = int(train_config['patience'])  # 确保是整数
                if patience_counter >= patience:
                    logger.info(f"早停在第 {epoch+1} 轮")
                    break
        
        training_time = time.time() - start_time
        return training_time
    
    def test_model_on_dataset(self, model_name: str, model_config: Dict[str, Any], 
                             dataset_name: str, dataset_config: Dict[str, Any]) -> TestResult:
        """在单个数据集上测试模型"""
        logger.info(f"\n测试 {model_name} 在 {dataset_name} 数据集上")
        
        result = TestResult(
            model_name=model_name,
            model_type=model_config.get('model_type', model_config.get('class', 'unknown')),
            dataset_name=dataset_name
        )
        
        try:
            # 获取数据配置
            data_config = self.config['data']
            batch_size = data_config['batch_size']
            
            # 生成数据
            train_data = self.generate_dataset(dataset_config, batch_size)
            val_data = self.generate_dataset(dataset_config, batch_size // 2)
            test_data = self.generate_dataset(dataset_config, batch_size // 4)
            
            # 计算维度
            input_dim = train_data[0].shape[1]
            output_dim = train_data[1].shape[1]
            
            # 创建模型
            model = self.model_factory.create_model(model_name, model_config, input_dim, output_dim)
            model = model.to(self.device)
            
            # 计算参数量
            param_count = self.count_parameters(model)
            result.param_count = param_count
            
            logger.info(f"模型参数量: {param_count:,}")
            
            # 训练模型
            training_time = self.train_model(model, train_data, val_data)
            result.training_time = training_time
            
            # 测试模型
            model.eval()
            test_x, test_y = test_data
            
            with torch.no_grad():
                start_time = time.time()
                
                if hasattr(model, 'forward') and 'time_steps' in str(model.forward.__code__.co_varnames):
                    time_steps = torch.zeros(test_x.shape[0], 1, dtype=torch.long).to(self.device)
                    predictions = model(test_x, time_steps)
                else:
                    predictions = model(test_x)
                
                inference_time = time.time() - start_time
                result.inference_time = inference_time
            
            # 计算指标
            metrics = self.calculate_metrics(predictions, test_y)
            result.mse = metrics['mse']
            result.mae = metrics['mae']
            result.rmse = metrics['rmse']
            result.relative_error = metrics['relative_error']
            result.r2_score = metrics['r2_score']
            
            # 形状信息
            result.input_shape = list(test_x.shape)
            result.output_shape = list(predictions.shape)
            result.expected_output_shape = list(test_y.shape)
            result.shape_match = predictions.shape == test_y.shape
            
            # 内存使用
            if torch.cuda.is_available():
                result.memory_usage_mb = torch.cuda.memory_allocated() / 1024**2
            
            result.success = True
            
            logger.info(f"测试成功 - MSE: {result.mse:.6f}, 推理时间: {result.inference_time:.4f}s")
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
            result.success = False
            result.error = str(e)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return result
    
    def run_all_tests(self) -> List[TestResult]:
        """运行所有测试"""
        logger.info("开始统一网络架构对比测试")
        
        models_config = self.config['models']
        datasets_config = self.config['data']['datasets']
        
        results = []
        
        for dataset_config in datasets_config:
            dataset_name = dataset_config['name']
            logger.info(f"\n处理数据集: {dataset_name}")
            
            for model_name, model_config in models_config.items():
                # 跳过不可用的模型
                model_type = model_config.get('model_type', model_config.get('class', ''))
                if 'transformer' in model_type.lower() and not TRANSFORMER_AVAILABLE:
                    logger.warning(f"跳过 {model_name}: Transformer不可用")
                    continue
                
                result = self.test_model_on_dataset(model_name, model_config, dataset_name, dataset_config)
                results.append(result)
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
    
    def generate_report(self, results: List[TestResult]) -> str:
        """生成测试报告"""
        report = ["\n" + "="*100]
        report.append("统一网络架构对比测试报告")
        report.append("="*100)
        report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        report.append(f"\n总测试数: {len(results)}")
        report.append(f"成功测试数: {len(successful_results)}")
        report.append(f"失败测试数: {len(failed_results)}")
        report.append(f"成功率: {len(successful_results)/len(results)*100:.1f}%")
        
        # 按数据集分组统计
        datasets = set(r.dataset_name for r in results)
        for dataset in datasets:
            dataset_results = [r for r in successful_results if r.dataset_name == dataset]
            if dataset_results:
                report.append(f"\n数据集 {dataset} 结果:")
                report.append("-" * 80)
                
                # 按MSE排序
                dataset_results.sort(key=lambda x: x.mse)
                
                for i, result in enumerate(dataset_results, 1):
                    report.append(f"{i}. {result.model_name} ({result.model_type})")
                    report.append(f"   MSE: {result.mse:.6f} | MAE: {result.mae:.6f} | R²: {result.r2_score:.4f}")
                    report.append(f"   参数量: {result.param_count:,} | 训练时间: {result.training_time:.2f}s | 推理时间: {result.inference_time:.4f}s")
        
        # 整体性能排名
        if successful_results:
            report.append("\n整体性能排名 (按平均MSE):")
            report.append("-" * 80)
            
            # 计算每个模型的平均性能
            model_stats = {}
            for result in successful_results:
                if result.model_name not in model_stats:
                    model_stats[result.model_name] = {
                        'mse_list': [],
                        'mae_list': [],
                        'r2_list': [],
                        'param_count': result.param_count,
                        'avg_training_time': 0,
                        'avg_inference_time': 0,
                        'model_type': result.model_type
                    }
                
                stats = model_stats[result.model_name]
                stats['mse_list'].append(result.mse)
                stats['mae_list'].append(result.mae)
                stats['r2_list'].append(result.r2_score)
                stats['avg_training_time'] += result.training_time
                stats['avg_inference_time'] += result.inference_time
            
            # 计算平均值
            for model_name, stats in model_stats.items():
                num_tests = len(stats['mse_list'])
                stats['avg_mse'] = np.mean(stats['mse_list'])
                stats['avg_mae'] = np.mean(stats['mae_list'])
                stats['avg_r2'] = np.mean(stats['r2_list'])
                stats['avg_training_time'] /= num_tests
                stats['avg_inference_time'] /= num_tests
            
            # 按平均MSE排序
            sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['avg_mse'])
            
            for i, (model_name, stats) in enumerate(sorted_models, 1):
                report.append(f"{i}. {model_name} ({stats['model_type']})")
                report.append(f"   平均MSE: {stats['avg_mse']:.6f} | 平均MAE: {stats['avg_mae']:.6f} | 平均R²: {stats['avg_r2']:.4f}")
                report.append(f"   参数量: {stats['param_count']:,} | 平均训练时间: {stats['avg_training_time']:.2f}s | 平均推理时间: {stats['avg_inference_time']:.4f}s")
        
        # 失败测试详情
        if failed_results:
            report.append("\n失败测试详情:")
            report.append("-" * 80)
            
            for result in failed_results:
                report.append(f"模型: {result.model_name} | 数据集: {result.dataset_name}")
                report.append(f"错误: {result.error}")
                report.append("")
        
        report.append("\n" + "="*100)
        
        return "\n".join(report)
    
    def save_results(self, results: List[TestResult], report: str):
        """保存测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        results_data = []
        for result in results:
            results_data.append({
                'model_name': result.model_name,
                'model_type': result.model_type,
                'dataset_name': result.dataset_name,
                'success': result.success,
                'error': result.error,
                'mse': result.mse,
                'mae': result.mae,
                'rmse': result.rmse,
                'relative_error': result.relative_error,
                'r2_score': result.r2_score,
                'training_time': result.training_time,
                'inference_time': result.inference_time,
                'memory_usage_mb': result.memory_usage_mb,
                'param_count': result.param_count,
                'input_shape': result.input_shape,
                'output_shape': result.output_shape,
                'expected_output_shape': result.expected_output_shape,
                'shape_match': result.shape_match
            })
        
        # 保存YAML格式
        results_file = Path(__file__).parent / f"unified_comparison_results_{timestamp}.yaml"
        with open(results_file, 'w', encoding='utf-8') as f:
            yaml.dump(results_data, f, default_flow_style=False, allow_unicode=True)
        
        # 保存JSON格式
        json_file = Path(__file__).parent / f"unified_comparison_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 保存报告
        report_file = Path(__file__).parent / f"unified_comparison_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"结果已保存到:")
        logger.info(f"  YAML: {results_file}")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  报告: {report_file}")

def main():
    """主函数"""
    config_path = Path(__file__).parent / "unified_comparison_config.yaml"
    
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    tester = UnifiedComparisonTester(str(config_path))
    
    try:
        # 运行测试
        results = tester.run_all_tests()
        
        # 生成报告
        report = tester.generate_report(results)
        print(report)
        
        # 保存结果
        tester.save_results(results, report)
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()