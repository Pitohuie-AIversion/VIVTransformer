#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态配置模型横向对比测试脚本

功能:
1. 使用dynamic_config.yaml的配置设定
2. 集成DynamicResolutionDataset进行数据处理
3. 对models目录中的所有增强模型进行横向对比
4. 支持动态输入输出分辨率配置
5. 生成详细的性能对比报告

作者: AI Assistant
日期: 2025
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置无头模式环境变量
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

# 添加路径
project_root = Path(__file__).parent.parent.parent
generate_data_path = project_root / 'generate_data'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(generate_data_path))
sys.path.insert(0, str(Path(__file__).parent))

# 导入dynamic_resolution_trainer中的数据集类
try:
    from dynamic_resolution_trainer import DynamicResolutionDataset, validate_config, log_gpu_memory, cleanup_memory
    HAS_DYNAMIC_TRAINER = True
    print("✅ 成功导入动态分辨率训练器")
except ImportError as e:
    HAS_DYNAMIC_TRAINER = False
    print(f"⚠️ 动态分辨率训练器导入失败: {e}")
    print("将使用简化的数据处理方式")

# 导入所有增强模型
from enhanced_fno import create_enhanced_fno1d, create_enhanced_fno2d
from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d
from enhanced_pinn import create_enhanced_pinn1d, create_enhanced_pinn2d
from enhanced_unet import create_enhanced_unet1d, create_enhanced_unet2d
from enhanced_transformer import create_enhanced_transformer1d, create_enhanced_transformer2d

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicModelsComparison:
    """
    动态配置模型横向对比测试器
    """
    
    def __init__(self, config_path: str, device='auto'):
        """
        初始化动态模型对比测试器
        
        Args:
            config_path: dynamic_config.yaml配置文件路径
            device: 计算设备
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"配置文件: {self.config_path}")
        
        # 从配置中提取关键参数
        self.data_config = self.config.get('data', {})
        self.input_resolution = tuple(self.data_config.get('input_resolution', [32, 32]))
        self.output_resolution = tuple(self.data_config.get('output_resolution', [128, 128]))
        self.batch_size = self.data_config.get('batch_size', 1)
        self.num_samples = self.data_config.get('num_samples', 100)
        
        logger.info(f"输入分辨率: {self.input_resolution}")
        logger.info(f"输出分辨率: {self.output_resolution}")
        logger.info(f"批次大小: {self.batch_size}")
        logger.info(f"样本数量: {self.num_samples}")
        
        # 计算维度
        self.input_dim = self.input_resolution[0] * self.input_resolution[1]
        self.output_dim = self.output_resolution[0] * self.output_resolution[1]
        
        logger.info(f"输入维度: {self.input_dim}")
        logger.info(f"输出维度: {self.output_dim}")
        
        # 测试结果存储
        self.test_results = {}
        
        # 模型配置 - 根据dynamic配置调整
        self.model_configs = self._setup_model_configs()
        
        # 初始化数据集
        self.dataset = None
        self.dataloader = None
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("✅ 配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def _setup_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """根据dynamic配置设置模型参数"""
        configs = {
            'FNO1d': {
                'creator': create_enhanced_fno1d,
                'base_params': {
                    'num_channels': 1, 
                    'modes': 16, 
                    'width': 64,
                    'input_resolution': self.input_resolution[0],
                    'output_resolution': self.output_resolution[0] * self.output_resolution[1]  # 128*128=16384
                },
                'dimensions': 1
            },
            'FNO2d': {
                'creator': create_enhanced_fno2d,
                'base_params': {
                    'num_channels': 1, 
                    'modes1': 12, 
                    'modes2': 12, 
                    'width': 20,
                    'input_resolution': self.input_resolution,
                    'output_resolution': self.output_resolution
                },
                'dimensions': 2
            },
            'MLP1d': {
                'creator': create_enhanced_mlp1d,
                'base_params': {
                    'input_channels': 1, 
                    'hidden_dim': 256, 
                    'num_layers': 6,
                    'input_resolution': self.input_resolution[0],
                    'output_resolution': self.output_resolution[0] * self.output_resolution[1]  # 扁平化输出: 128*128=16384
                },
                'dimensions': 1
            },
            'MLP2d': {
                'creator': create_enhanced_mlp2d,
                'base_params': {
                    'input_channels': 1, 
                    'hidden_dim': 256, 
                    'num_layers': 6,
                    'input_resolution': self.input_resolution,
                    'output_resolution': self.output_resolution
                },
                'dimensions': 2
            },
            'PINN1d': {
                'creator': create_enhanced_pinn1d,
                'base_params': {
                    'output_dim': 1, 
                    'hidden_dim': 256, 
                    'num_layers': 6,
                    'input_resolution': self.input_resolution[0],
                    'output_resolution': self.output_resolution[0] * self.output_resolution[1]  # 扁平化输出: 128*128=16384
                },
                'dimensions': 1
            },
            'PINN2d': {
                'creator': create_enhanced_pinn2d,
                'base_params': {
                    'output_dim': 1,  # 保持单输出维度，在forward中处理形状
                    'hidden_dim': 256, 
                    'num_layers': 6,
                    'input_resolution': self.input_resolution,
                    'output_resolution': self.output_resolution[0] * self.output_resolution[1]  # 扁平化输出: 128*128=16384
                },
                'dimensions': 2
            },
            'UNet1d': {
                'creator': create_enhanced_unet1d,
                'base_params': {
                    'in_channels': 1, 
                    'out_channels': 1, 
                    'init_features': 32,
                    'input_resolution': self.input_resolution[0],
                    'output_resolution': self.output_resolution[0] * self.output_resolution[1]  # 扁平化输出: 128*128=16384
                },
                'dimensions': 1
            },
            'UNet2d': {
                'creator': create_enhanced_unet2d,
                'base_params': {
                    'in_channels': 1, 
                    'out_channels': 1, 
                    'init_features': 32,
                    'input_resolution': self.input_resolution,
                    'output_resolution': self.output_resolution
                },
                'dimensions': 2
            },
            'Transformer1d': {
                'creator': create_enhanced_transformer1d,
                'base_params': {
                    'input_channels': 1,
                    'output_channels': 1,
                    'd_model': 128,
                    'num_heads': 4,
                    'num_layers': 3,
                    'input_resolution': self.input_resolution[0],
                    'output_resolution': self.output_resolution[0] * self.output_resolution[1],  # 扁平化输出: 128*128=16384
                    'attention_type': 'simplified_self_attention',
                    'pe_type': 'learnable_1d'
                },
                'dimensions': 1
            },
            'Transformer2d': {
                'creator': create_enhanced_transformer2d,
                'base_params': {
                    'input_channels': 1,
                    'output_channels': 1,
                    'd_model': 128,
                    'num_heads': 4,
                    'num_layers': 3,
                    'input_resolution': self.input_resolution,
                    'output_resolution': self.output_resolution,
                    'attention_type': 'simplified_self_attention',
                    'pe_type': 'learnable_2d'
                },
                'dimensions': 2
            }
        }
        
        logger.info(f"配置了 {len(configs)} 个模型")
        return configs
    
    def setup_dataset(self):
        """设置数据集"""
        data_path = self.data_config.get('path')
        if not data_path or not Path(data_path).exists():
            logger.error(f"数据文件不存在: {data_path}")
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        if HAS_DYNAMIC_TRAINER:
            # 使用动态分辨率数据集
            self.dataset = DynamicResolutionDataset(
                data_path=data_path,
                input_resolution=self.input_resolution,
                output_resolution=self.output_resolution,
                num_samples=self.num_samples,
                crop_mode=self.data_config.get('crop_mode', 'center'),
                normalize_data=self.data_config.get('normalize_data', True),
                lazy_loading=self.data_config.get('lazy_loading', False)
            )
            logger.info("✅ 使用DynamicResolutionDataset")
        else:
            # 使用简化的数据集
            self.dataset = self._create_simple_dataset(data_path)
            logger.info("✅ 使用简化数据集")
        
        # 创建数据加载器
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # 避免多进程问题
            pin_memory=False
        )
        
        logger.info(f"数据集大小: {len(self.dataset)}")
        logger.info(f"数据加载器批次数: {len(self.dataloader)}")
    
    def _create_simple_dataset(self, data_path: str):
        """创建简化的数据集（当DynamicResolutionDataset不可用时）"""
        import h5py
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data_path, input_res, output_res, num_samples):
                self.data_path = data_path
                self.input_res = input_res
                self.output_res = output_res
                self.num_samples = num_samples
                
                # 加载数据
                with h5py.File(data_path, 'r') as f:
                    data = np.array(f['tensor'][:num_samples], dtype=np.float32)
                    if len(data.shape) == 4 and data.shape[1] == 1:
                        data = data.squeeze(1)
                    self.data = data
                
                logger.info(f"简化数据集加载完成，形状: {self.data.shape}")
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                
                # 简单的裁剪到输入分辨率
                h, w = sample.shape
                ih, iw = self.input_res
                oh, ow = self.output_res
                
                # 中心裁剪到输入分辨率
                start_h = (h - ih) // 2
                start_w = (w - iw) // 2
                input_data = sample[start_h:start_h+ih, start_w:start_w+iw]
                
                # 中心裁剪到输出分辨率
                start_h = (h - oh) // 2
                start_w = (w - ow) // 2
                output_data = sample[start_h:start_h+oh, start_w:start_w+ow]
                
                # 转换为tensor并添加通道维度
                input_tensor = torch.from_numpy(input_data).unsqueeze(0).float()
                output_tensor = torch.from_numpy(output_data).unsqueeze(0).float()
                
                return input_tensor, output_tensor
        
        return SimpleDataset(data_path, self.input_resolution, self.output_resolution, self.num_samples)
    
    def create_model(self, model_name: str) -> nn.Module:
        """创建指定的模型"""
        if model_name not in self.model_configs:
            raise ValueError(f"不支持的模型: {model_name}")
        
        config = self.model_configs[model_name]
        creator = config['creator']
        params = config['base_params'].copy()
        
        try:
            model = creator(**params)
            model = model.to(self.device)
            logger.info(f"✅ 成功创建模型: {model_name}")
            return model
        except Exception as e:
            logger.error(f"❌ 创建模型失败 {model_name}: {e}")
            raise
    
    def test_model(self, model_name: str, model: nn.Module, num_test_samples: int = 10) -> Dict[str, Any]:
        """测试单个模型"""
        model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        total_time = 0.0
        successful_tests = 0
        
        with torch.no_grad():
            for i, (input_data, target_data, _) in enumerate(self.dataloader):
                if i >= num_test_samples:
                    break
                
                try:
                    input_data = input_data.to(self.device)
                    target_data = target_data.to(self.device)
                    
                    # 根据模型维度要求重塑输入数据
                    model_config = self.model_configs.get(model_name, {})
                    model_dims = model_config.get('dimensions', 1)
                    
                    if model_dims == 2:
                        # 2D模型需要4D张量 [batch, height, width, channels]
                        if len(input_data.shape) == 2:  # [batch, flattened]
                            batch_size = input_data.shape[0]
                            h, w = self.input_resolution
                            input_data = input_data.view(batch_size, h, w, 1)
                    elif model_dims == 1:
                        # 1D模型需要3D张量 [batch, length, channels]
                        if len(input_data.shape) == 2:  # [batch, flattened]
                            batch_size = input_data.shape[0]
                            length = input_data.shape[1]
                            input_data = input_data.view(batch_size, length, 1)
                    
                    # 记录推理时间
                    start_time = time.time()
                    
                    # 前向传播
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        output = model(input_data)
                    
                    inference_time = time.time() - start_time
                    
                    # 确保输出形状匹配
                    if output.shape != target_data.shape:
                        # 尝试调整输出形状
                        if output.numel() == target_data.numel():
                            output = output.view(target_data.shape)
                        else:
                            logger.warning(f"形状不匹配: 输出{output.shape} vs 目标{target_data.shape}")
                            continue
                    
                    # 计算损失
                    mse = torch.mean((output - target_data) ** 2).item()
                    mae = torch.mean(torch.abs(output - target_data)).item()
                    
                    total_mse += mse
                    total_mae += mae
                    total_time += inference_time
                    successful_tests += 1
                    
                except Exception as e:
                    logger.warning(f"测试样本 {i} 失败: {e}")
                    continue
        
        if successful_tests == 0:
            return {
                'model_name': model_name,
                'success': False,
                'error': '所有测试样本都失败',
                'mse': float('inf'),
                'mae': float('inf'),
                'avg_inference_time': float('inf')
            }
        
        avg_mse = total_mse / successful_tests
        avg_mae = total_mae / successful_tests
        avg_time = total_time / successful_tests
        
        return {
            'model_name': model_name,
            'success': True,
            'mse': avg_mse,
            'mae': avg_mae,
            'avg_inference_time': avg_time,
            'successful_tests': successful_tests,
            'total_tests': num_test_samples
        }
    
    def run_comparison(self, num_test_samples: int = 10) -> Dict[str, Any]:
        """运行所有模型的横向对比测试"""
        logger.info("🚀 开始动态配置模型横向对比测试")
        
        # 设置数据集
        self.setup_dataset()
        
        # 记录GPU内存
        if HAS_DYNAMIC_TRAINER:
            log_gpu_memory("测试开始前")
        
        results = {
            'config_info': {
                'input_resolution': self.input_resolution,
                'output_resolution': self.output_resolution,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'batch_size': self.batch_size,
                'num_samples': self.num_samples,
                'device': str(self.device)
            },
            'model_results': {},
            'summary': {}
        }
        
        # 测试每个模型
        for model_name in self.model_configs.keys():
            logger.info(f"\n📊 测试模型: {model_name}")
            
            try:
                # 创建模型
                model = self.create_model(model_name)
                
                # 测试模型
                result = self.test_model(model_name, model, num_test_samples)
                results['model_results'][model_name] = result
                
                if result['success']:
                    logger.info(f"✅ {model_name} 测试成功")
                    logger.info(f"   MSE: {result['mse']:.6f}")
                    logger.info(f"   MAE: {result['mae']:.6f}")
                    logger.info(f"   推理时间: {result['avg_inference_time']:.4f}s")
                else:
                    logger.error(f"❌ {model_name} 测试失败: {result.get('error', '未知错误')}")
                
                # 清理内存
                del model
                if HAS_DYNAMIC_TRAINER:
                    cleanup_memory()
                
            except Exception as e:
                logger.error(f"❌ {model_name} 测试异常: {e}")
                results['model_results'][model_name] = {
                    'model_name': model_name,
                    'success': False,
                    'error': str(e),
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'avg_inference_time': float('inf')
                }
        
        # 生成汇总统计
        results['summary'] = self._generate_summary(results['model_results'])
        
        # 记录最终GPU内存
        if HAS_DYNAMIC_TRAINER:
            log_gpu_memory("测试完成后")
        
        logger.info("\n🎉 动态配置模型横向对比测试完成")
        return results
    
    def _generate_summary(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成测试结果汇总"""
        successful_models = [name for name, result in model_results.items() if result['success']]
        failed_models = [name for name, result in model_results.items() if not result['success']]
        
        if not successful_models:
            return {
                'total_models': len(model_results),
                'successful_models': 0,
                'failed_models': len(failed_models),
                'success_rate': 0.0,
                'best_mse_model': None,
                'best_speed_model': None
            }
        
        # 找到最佳性能模型
        best_mse_model = min(successful_models, key=lambda x: model_results[x]['mse'])
        best_speed_model = min(successful_models, key=lambda x: model_results[x]['avg_inference_time'])
        
        # 计算平均性能
        avg_mse = np.mean([model_results[name]['mse'] for name in successful_models])
        avg_mae = np.mean([model_results[name]['mae'] for name in successful_models])
        avg_time = np.mean([model_results[name]['avg_inference_time'] for name in successful_models])
        
        return {
            'total_models': len(model_results),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'success_rate': len(successful_models) / len(model_results) * 100,
            'successful_model_list': successful_models,
            'failed_model_list': failed_models,
            'best_mse_model': best_mse_model,
            'best_mse_value': model_results[best_mse_model]['mse'],
            'best_speed_model': best_speed_model,
            'best_speed_value': model_results[best_speed_model]['avg_inference_time'],
            'average_performance': {
                'mse': avg_mse,
                'mae': avg_mae,
                'inference_time': avg_time
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str = "dynamic_models_comparison_results.txt"):
        """保存测试结果到文件"""
        output_file = Path(output_path)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("动态配置模型横向对比测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 配置信息
            f.write("配置信息:\n")
            f.write("-" * 20 + "\n")
            config_info = results['config_info']
            f.write(f"输入分辨率: {config_info['input_resolution']}\n")
            f.write(f"输出分辨率: {config_info['output_resolution']}\n")
            f.write(f"输入维度: {config_info['input_dim']}\n")
            f.write(f"输出维度: {config_info['output_dim']}\n")
            f.write(f"批次大小: {config_info['batch_size']}\n")
            f.write(f"样本数量: {config_info['num_samples']}\n")
            f.write(f"计算设备: {config_info['device']}\n\n")
            
            # 汇总统计
            f.write("测试汇总:\n")
            f.write("-" * 20 + "\n")
            summary = results['summary']
            f.write(f"总模型数: {summary['total_models']}\n")
            f.write(f"成功模型数: {summary['successful_models']}\n")
            f.write(f"失败模型数: {summary['failed_models']}\n")
            f.write(f"成功率: {summary['success_rate']:.1f}%\n")
            
            if summary['successful_models'] > 0:
                f.write(f"最佳精度模型: {summary['best_mse_model']} (MSE: {summary['best_mse_value']:.6f})\n")
                f.write(f"最快推理模型: {summary['best_speed_model']} (时间: {summary['best_speed_value']:.4f}s)\n")
                
                avg_perf = summary['average_performance']
                f.write(f"平均MSE: {avg_perf['mse']:.6f}\n")
                f.write(f"平均MAE: {avg_perf['mae']:.6f}\n")
                f.write(f"平均推理时间: {avg_perf['inference_time']:.4f}s\n")
            
            f.write("\n")
            
            # 详细结果
            f.write("详细测试结果:\n")
            f.write("-" * 20 + "\n")
            
            for model_name, result in results['model_results'].items():
                f.write(f"\n模型: {model_name}\n")
                if result['success']:
                    f.write(f"  状态: 成功\n")
                    f.write(f"  MSE: {result['mse']:.6f}\n")
                    f.write(f"  MAE: {result['mae']:.6f}\n")
                    f.write(f"  平均推理时间: {result['avg_inference_time']:.4f}s\n")
                    f.write(f"  成功测试数: {result['successful_tests']}/{result['total_tests']}\n")
                else:
                    f.write(f"  状态: 失败\n")
                    f.write(f"  错误: {result.get('error', '未知错误')}\n")
        
        logger.info(f"✅ 测试结果已保存到: {output_file}")

def main():
    """主函数"""
    # 配置文件路径
    config_path = "../../generate_data/dynamic_config.yaml"
    
    if not Path(config_path).exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    try:
        # 创建测试器
        tester = DynamicModelsComparison(config_path)
        
        # 运行对比测试
        results = tester.run_comparison(num_test_samples=20)
        
        # 保存结果
        tester.save_results(results)
        
        # 打印汇总
        summary = results['summary']
        print("\n" + "=" * 50)
        print("动态配置模型横向对比测试完成")
        print("=" * 50)
        print(f"成功率: {summary['success_rate']:.1f}% ({summary['successful_models']}/{summary['total_models']})")
        
        if summary['successful_models'] > 0:
            print(f"最佳精度: {summary['best_mse_model']} (MSE: {summary['best_mse_value']:.6f})")
            print(f"最快推理: {summary['best_speed_model']} (时间: {summary['best_speed_value']:.4f}s)")
        
        print("详细结果已保存到: dynamic_models_comparison_results.txt")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()