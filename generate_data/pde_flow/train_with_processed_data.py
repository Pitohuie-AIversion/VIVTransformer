#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用处理后的PDEBench数据进行训练的脚本

功能:
1. 加载之前处理的PDEBench数据
2. 配置训练参数
3. 启动训练流程
4. 保存训练结果

作者: AI Assistant
日期: 2025
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
import argparse
import yaml

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / 'modify_multi_attention'))
sys.path.append(str(Path(__file__).parent))

# 导入训练相关模块
from data.dataloader import get_loaders
from mymodels.transformer import TransformerFlowReconstructionModel
from training.trainer import train_model, test_model
from utils.visualization import plot_losses
from utils.svd10_loss import TotalLossWithSVD
from utils.config import load_config
from utils.logging_utils import setup_logging

# 导入数据处理模块
from pdebench_data_processor import PDEBenchProcessor, PDEBenchConfig
from config_loader import ConfigLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_training_config(args=None, config_file=None):
    """
    创建训练配置，支持软编码
    
    Args:
        args: 命令行参数对象
        config_file: 配置文件路径
    """
    # 默认配置
    default_config = {
        'data': {
            'path': 'input32_output128_dataset.npz',
            'batch_size': 16,
            'dataset_type': 'processed_pdebench',
            'max_samples': None,
            'train_ratio': 0.7,
            'valid_ratio': 0.15,
            'test_ratio': 0.15
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 15,
            'min_delta': 1e-6,
            'save_best_model': True,
            'model_save_path': './models/best_model_processed_pdebench.pth',
            'log_interval': 10
        },
        'model': {
            'input_dim': 1024,  # 32x32 = 1024
            'output_dim': 16384,  # 128x128 = 16384
            'num_layers': 4,
            'd_model': 512,
            'num_heads': 8,
            'max_time_steps': 100,
            'seq_len': 32,  # sqrt(1024) = 32
            'dropout': 0.1
        },
        'attention': {
            'type': 'sge',
            'available_types': ['sge', 'cbam', 'eca', 'se', 'relative', 'external']
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': './logs/training_processed_pdebench.log'
        },
        'device': 'auto',  # 'auto', 'cuda', 'cpu'
        'seed': 42,
        'visualization': {
            'enabled': True,
            'interval': 10,
            'max_samples': 5
        }
    }
    
    config = default_config.copy()
    
    # 从配置文件加载（如果提供）
    if config_file and Path(config_file).exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    import yaml
                    file_config = yaml.safe_load(f)
                else:
                    import json
                    file_config = json.load(f)
            
            # 递归更新配置
            def update_config(base, update):
                for key, value in update.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        update_config(base[key], value)
                    else:
                        base[key] = value
            
            update_config(config, file_config)
            logger.info(f"从配置文件加载配置: {config_file}")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
    
    # 从命令行参数覆盖（如果提供）
    if args:
        if hasattr(args, 'epochs') and args.epochs:
            config['training']['epochs'] = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size:
            config['data']['batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
        if hasattr(args, 'attention_type') and args.attention_type:
            config['attention']['type'] = args.attention_type
        if hasattr(args, 'device') and args.device:
            config['device'] = args.device
        if hasattr(args, 'data_file') and args.data_file:
            config['data']['path'] = args.data_file
        if hasattr(args, 'input_dim') and args.input_dim:
            config['model']['input_dim'] = args.input_dim
            config['model']['output_dim'] = args.input_dim  # 通常输入输出维度相同
            # 自动计算seq_len
            import math
            seq_len = int(math.sqrt(args.input_dim))
            if seq_len * seq_len == args.input_dim:
                config['model']['seq_len'] = seq_len
            else:
                logger.warning(f"输入维度 {args.input_dim} 不是完全平方数，使用默认seq_len")
    
    # 设备自动检测
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config

def prepare_processed_data():
    """
    准备处理后的数据用于训练
    """
    logger.info("=== 准备处理后的数据 ===")
    
    # 检查是否存在处理后的数据
    processed_files = [
        'spatial_crop_32x32_demo.npz',
        'pdebench_processed_data.pt'
    ]
    
    available_files = []
    for file in processed_files:
        if Path(file).exists():
            available_files.append(file)
            logger.info(f"找到处理后的数据: {file}")
    
    if not available_files:
        logger.info("未找到处理后的数据，开始处理原始数据...")
        
        # 使用PDEBench处理器生成数据
        config = PDEBenchConfig()
        config.PDEBENCH_ROOT = r"X:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download"
        config.MAX_SAMPLES = 1000
        config.SPATIAL_CROP = ((48, 80), (48, 80))  # 32x32裁剪
        config.FLATTEN_SPATIAL = True
        config.OUTPUT_FILE = 'processed_pdebench_for_training.pt'
        
        processor = PDEBenchProcessor(config)
        success = processor.process_pde_dataset('darcy', sequence_length=1)
        
        if success:
            processor.save_processed_data()
            return config.OUTPUT_FILE
        else:
            raise RuntimeError("数据处理失败")
    else:
        return available_files[0]

class ProcessedPDEBenchDataset(torch.utils.data.Dataset):
    """
    处理后的PDEBench数据集类
    """
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = None
        self.inputs = None
        self.outputs = None
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        if self.data_path.suffix == '.npz':
            # NumPy格式
            data = np.load(self.data_path)
            if 'inputs' in data and 'outputs' in data:
                # 新格式：输入32x32，输出128x128
                self.inputs = data['inputs']  # [samples, 1024] (32x32)
                self.outputs = data['outputs']  # [samples, 16384] (128x128)
                logger.info(f"加载新格式数据: 输入{data.get('input_shape', 'unknown')}，输出{data.get('output_shape', 'unknown')}")
            elif 'cropped' in data:
                # 旧格式：空间裁剪数据
                cropped_data = data['cropped']  # [samples, height, width]
                # 展平空间维度
                self.inputs = cropped_data.reshape(cropped_data.shape[0], -1)
                # 创建简单的预测任务：预测下一个样本
                self.outputs = np.roll(self.inputs, -1, axis=0)
                self.outputs[-1] = self.inputs[-1]  # 最后一个样本预测自己
                logger.info("加载旧格式裁剪数据")
            else:
                raise ValueError("NPZ文件中未找到'inputs'和'outputs'或'cropped'数据")
                
        elif self.data_path.suffix == '.pt':
            # PyTorch格式
            data = torch.load(self.data_path)
            if isinstance(data, dict):
                if 'darcy' in data:
                    pde_data = data['darcy']
                    self.inputs = pde_data['inputs']
                    self.outputs = pde_data['outputs']
                else:
                    # 假设是简单的张量字典
                    keys = list(data.keys())
                    if len(keys) >= 2:
                        self.inputs = data[keys[0]]
                        self.outputs = data[keys[1]]
                    else:
                        raise ValueError("PyTorch文件格式不支持")
            else:
                # 假设是单个张量
                self.inputs = data[:-1]  # 除最后一个外的所有样本
                self.outputs = data[1:]   # 除第一个外的所有样本
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")
        
        # 转换为PyTorch张量
        if not isinstance(self.inputs, torch.Tensor):
            self.inputs = torch.FloatTensor(self.inputs)
        if not isinstance(self.outputs, torch.Tensor):
            self.outputs = torch.FloatTensor(self.outputs)
            
        logger.info(f"加载数据: 输入形状 {self.inputs.shape}, 输出形状 {self.outputs.shape}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # 返回 (input, output, time_step)
        return self.inputs[idx], self.outputs[idx], torch.tensor(idx, dtype=torch.float32)
    
    def get_data_statistics(self):
        """获取数据统计信息"""
        return {
            'num_samples': len(self),
            'input_shape': self.inputs.shape,
            'output_shape': self.outputs.shape,
            'input_range': (self.inputs.min().item(), self.inputs.max().item()),
            'output_range': (self.outputs.min().item(), self.outputs.max().item())
        }

def get_processed_loaders(data_path, batch_size, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """
    获取处理后数据的加载器
    """
    dataset = ProcessedPDEBenchDataset(data_path)
    
    # 打印数据统计信息
    stats = dataset.get_data_statistics()
    logger.info(f"数据集统计: {stats}")
    
    # 分割数据集
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}, 测试集: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader

def main():
    """
    主训练函数
    """
    parser = argparse.ArgumentParser(description='使用处理后的PDEBench数据进行训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径 (YAML/JSON)')
    parser.add_argument('--data_file', type=str, default=None, help='处理后的数据文件路径')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--attention_type', type=str, default=None, help='注意力机制类型')
    parser.add_argument('--device', type=str, default=None, help='设备类型 (auto/cuda/cpu)')
    parser.add_argument('--input_dim', type=int, default=None, help='输入维度')
    parser.add_argument('--num_layers', type=int, default=None, help='模型层数')
    parser.add_argument('--d_model', type=int, default=None, help='模型隐藏维度')
    parser.add_argument('--num_heads', type=int, default=None, help='注意力头数')
    
    args = parser.parse_args()
    
    # 1. 创建配置
    config = create_training_config(args, args.config)
    
    # 从配置中获取设备
    device = config['device']
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 添加命令行参数覆盖
    if hasattr(args, 'num_layers') and args.num_layers:
        config['model']['num_layers'] = args.num_layers
    if hasattr(args, 'd_model') and args.d_model:
        config['model']['d_model'] = args.d_model
    if hasattr(args, 'num_heads') and args.num_heads:
        config['model']['num_heads'] = args.num_heads
    
    try:
        # 2. 准备数据
        data_file = config['data']['path']
        if not Path(data_file).exists():
            logger.info(f"数据文件 {data_file} 不存在，开始处理数据...")
            data_file = prepare_processed_data()
        
        logger.info(f"使用数据文件: {data_file}")
        
        # 3. 创建数据加载器
        train_loader, valid_loader, test_loader = get_processed_loaders(
            data_file, 
            config['data']['batch_size'],
            config['data']['train_ratio'],
            config['data']['valid_ratio'],
            config['data']['test_ratio']
        )
        
        # 4. 创建模型
        model = TransformerFlowReconstructionModel(
            input_dim=config['model']['input_dim'],
            output_dim=config['model']['output_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            d_model=config['model']['d_model'],
            max_time_steps=config['model']['max_time_steps'],
            attention_type=config['attention']['type'],
            seq_len=config['model']['seq_len']
        )
        
        logger.info(f"创建模型: {model.__class__.__name__}")
        logger.info(f"注意力机制: {config['attention']['type']}")
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"配置信息: 输入维度={config['model']['input_dim']}, 序列长度={config['model']['seq_len']}, 隐藏维度={config['model']['d_model']}")
        
        # 5. 设置优化器和损失函数
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'], 
            weight_decay=config['training']['weight_decay']
        )
        
        criterion = TotalLossWithSVD()
        
        # 6. 创建结果目录
        result_dir = f"training_results_{config['attention']['type']}"
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(f"{result_dir}/models", exist_ok=True)
        os.makedirs(f"{result_dir}/logs", exist_ok=True)
        
        # 保存配置文件
        config_save_path = f"{result_dir}/config_used.yaml"
        try:
            import yaml
            with open(config_save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置已保存到: {config_save_path}")
        except Exception as e:
            logger.warning(f"保存配置文件失败: {e}")
        
        # 7. 开始训练
        logger.info("=== 开始训练 ===")
        train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config['training']['epochs'],
            device=device,
            early_stop_patience=config['training']['patience'],
            attention_type=config['attention']['type'],
            result_dir=result_dir,
            cfg=config
        )
        
        # 8. 测试模型
        logger.info("=== 开始测试 ===")
        test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            attention_type=config['attention']['type'],
            parent_dir=result_dir,
            cfg=config
        )
        
        logger.info("=== 训练完成 ===")
        logger.info(f"结果保存在: {result_dir}")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()