#usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_model_config.yaml 专用测试运行脚本

功能:
1. 使用multi_model_config.yaml配置文件
2. 支持所有配置的模型类型（Transformer、MLP、FNO、UNet）
3. 集成模型工厂和数据处理流程
4. 生成详细的测试报告

作者: AI Assistant
日期: 2025
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime
import traceback

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'generate_data'))
sys.path.insert(0, str(project_root / 'modify_multi_attention'))

# 导入必要的模块
try:
    from models.unified_model_factory import UnifiedModelFactory
    from models.unified_comparison_test import SyntheticDataGenerator, TestResult
    print("✅ 成功导入统一模型工厂")
except ImportError as e:
    print(f"⚠️  导入统一模型工厂失败: {e}")
    UnifiedModelFactory = None

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_model_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class MultiModelTester:
    """多模型测试器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = self.setup_device()
        self.data_generator = None
        self.model_factory = None
        self.results = {}
        
        # 初始化组件
        self.setup_components()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("✅ 配置文件加载成功")
            return config
        except Exception as e:
            logger.error(f"❌ 配置文件加载失败: {e}")
            raise
    
    def setup_device(self) -> torch.device:
        """设置设备"""
        device_config = self.config.get('device', {})
        use_cuda = device_config.get('use_cuda', True)
        device_id = device_config.get('device_id', 0)
        
        if use_cuda and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
            logger.info(f"✅ 使用GPU设备: {device}")
        else:
            device = torch.device('cpu')
            logger.info("✅ 使用CPU设备")
        
        return device
    
    def setup_components(self):
        """初始化组件"""
        # 初始化数据生成器
        self.data_generator = SyntheticDataGenerator(self.config)
        
        # 初始化模型工厂
        if UnifiedModelFactory:
            self.model_factory = UnifiedModelFactory()
        else:
            logger.warning("⚠️  统一模型工厂不可用，将使用简化的模型创建")
    
    def create_simple_transformer(self, config: Dict[str, Any]) -> nn.Module:
        """创建简化的Transformer模型"""
        input_dim = config.get('input_dim', 16384)
        output_dim = config.get('output_dim', 16384)
        d_model = config.get('d_model', 64)
        num_heads = config.get('num_heads', 4)
        num_layers = config.get('num_layers', 2)
        seq_len = config.get('seq_len', 256)
        
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                # 使用实际的序列长度64而不是配置中的16384
                actual_input_dim = 64
                actual_output_dim = 64
                self.input_projection = nn.Linear(actual_input_dim, d_model * seq_len)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=d_model * 4,
                        dropout=0.1,
                        batch_first=True
                    ),
                    num_layers=num_layers
                )
                self.output_projection = nn.Linear(d_model * seq_len, actual_output_dim)
            
            def forward(self, x):
                batch_size = x.size(0)
                x = self.input_projection(x)  # [batch, d_model * seq_len]
                x = x.view(batch_size, seq_len, d_model)  # [batch, seq_len, d_model]
                x = self.transformer(x)  # [batch, seq_len, d_model]
                x = x.view(batch_size, -1)  # [batch, d_model * seq_len]
                x = self.output_projection(x)  # [batch, actual_output_dim]
                return x
        
        return SimpleTransformer()
    
    def create_mlp(self, config: Dict[str, Any]) -> nn.Module:
        """创建MLP模型"""
        # 使用实际的序列长度作为输入维度
        input_dim = 64  # 实际数据的序列长度
        output_dim = 64  # 输出也是序列长度
        hidden_dims = config.get('hidden_dims', [128, 256, 128])
        activation = config.get('activation', 'gelu')
        dropout = config.get('dropout', 0.1)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation.lower() == 'gelu':
                layers.append(nn.GELU())
            elif activation.lower() == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def create_simple_fno(self, config: Dict[str, Any]) -> nn.Module:
        """创建简化的FNO模型（使用MLP近似）"""
        input_dim = 64  # 实际序列长度
        output_dim = 64  # 实际序列长度
        width = config.get('width', 64)
        num_layers = config.get('num_layers', 4)
        
        # 使用MLP近似FNO的行为
        layers = []
        layers.append(nn.Linear(input_dim, width * 4))
        layers.append(nn.GELU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(width * 4, width * 4))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(width * 4, output_dim))
        
        return nn.Sequential(*layers)
    
    def create_simple_unet(self, config: Dict[str, Any]) -> nn.Module:
        """创建简化的UNet模型（使用MLP近似）"""
        input_dim = 64  # 实际序列长度
        output_dim = 64  # 实际序列长度
        features = config.get('features', [32, 64, 128, 64, 32])
        
        # 使用MLP近似UNet的编码器-解码器结构
        layers = []
        prev_dim = input_dim
        
        # 编码器部分
        for feature in features[:len(features)//2 + 1]:
            layers.append(nn.Linear(prev_dim, feature * 16))
            layers.append(nn.ReLU())
            prev_dim = feature * 16
        
        # 解码器部分
        for feature in reversed(features[len(features)//2 + 1:]):
            layers.append(nn.Linear(prev_dim, feature * 16))
            layers.append(nn.ReLU())
            prev_dim = feature * 16
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def create_model(self, model_name: str, model_config: Dict[str, Any]) -> nn.Module:
        """创建模型"""
        model_type = model_config.get('model_type', '')
        
        try:
            # 首先尝试使用统一模型工厂
            if self.model_factory:
                try:
                    model = self.model_factory.create_model(model_config)
                    logger.info(f"✅ 使用统一模型工厂创建模型: {model_name}")
                    return model
                except Exception as e:
                    logger.warning(f"⚠️  统一模型工厂创建失败，使用简化模型: {e}")
            
            # 使用简化模型创建
            if model_type == 'transformer':
                model = self.create_simple_transformer(model_config)
            elif model_type == 'mlp':
                model = self.create_mlp(model_config)
            elif model_type == 'fno':
                model = self.create_simple_fno(model_config)
            elif model_type == 'unet':
                model = self.create_simple_unet(model_config)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            logger.info(f"✅ 使用简化方法创建模型: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"❌ 模型创建失败 {model_name}: {e}")
            raise
    
    def count_parameters(self, model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train_model(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor], 
                   model_name: str) -> Dict[str, Any]:
        """训练模型"""
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 3)
        learning_rate = training_config.get('learning_rate', 0.001)
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_input, train_target = train_data
        train_input = train_input.to(self.device)
        train_target = train_target.to(self.device)
        
        # 训练循环
        start_time = time.time()
        losses = []
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            try:
                output = model(train_input)
                loss = criterion(output, train_target)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
                if epoch % max(1, epochs // 5) == 0:
                    logger.info(f"  Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                logger.error(f"❌ 训练过程中出错 {model_name} epoch {epoch}: {e}")
                break
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'final_loss': losses[-1] if losses else float('inf'),
            'losses': losses
        }
    
    def evaluate_model(self, model: nn.Module, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        test_input, test_target = test_data
        test_input = test_input.to(self.device)
        test_target = test_target.to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            output = model(test_input)
            inference_time = time.time() - start_time
            
            # 计算指标
            mse = nn.MSELoss()(output, test_target).item()
            mae = nn.L1Loss()(output, test_target).item()
            
            # 计算R²
            ss_res = torch.sum((test_target - output) ** 2).item()
            ss_tot = torch.sum((test_target - torch.mean(test_target)) ** 2).item()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'inference_time': inference_time
            }
    
    def test_single_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """测试单个模型"""
        logger.info(f"\n🔧 测试模型: {model_name}")
        
        try:
            # 创建模型
            model = self.create_model(model_name, model_config)
            param_count = self.count_parameters(model)
            logger.info(f"  参数量: {param_count:,}")
            
            # 生成数据
            data_config = self.config.get('data', {})
            batch_size = data_config.get('batch_size', 2)
            
            train_data = self.data_generator.generate_time_series_data(
                batch_size=batch_size,
                seq_len=64
            )
            test_data = self.data_generator.generate_time_series_data(
                batch_size=batch_size,
                seq_len=64
            )
            
            # 训练模型
            logger.info("  开始训练...")
            train_results = self.train_model(model, train_data, model_name)
            
            # 评估模型
            logger.info("  开始评估...")
            eval_results = self.evaluate_model(model, test_data)
            
            # 汇总结果
            results = {
                'model_name': model_name,
                'model_type': model_config.get('model_type', ''),
                'parameters': param_count,
                'training_time': train_results['training_time'],
                'final_training_loss': train_results['final_loss'],
                'test_mse': eval_results['mse'],
                'test_mae': eval_results['mae'],
                'test_rmse': eval_results['rmse'],
                'test_r2': eval_results['r2'],
                'inference_time': eval_results['inference_time'],
                'status': 'success'
            }
            
            logger.info(f"  ✅ 测试成功 - MSE: {eval_results['mse']:.6f}, R²: {eval_results['r2']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"  ❌ 测试失败: {e}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            
            return {
                'model_name': model_name,
                'model_type': model_config.get('model_type', ''),
                'parameters': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有模型测试"""
        logger.info("🚀 开始多模型测试")
        logger.info("=" * 60)
        
        models = self.config.get('models', {})
        results = {}
        
        for model_name, model_config in models.items():
            results[model_name] = self.test_single_model(model_name, model_config)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"multi_model_test_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("多模型测试报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置文件: {self.config_path}\n\n")
            
            # 统计信息
            total_models = len(results)
            successful_models = sum(1 for r in results.values() if r.get('status') == 'success')
            failed_models = total_models - successful_models
            
            f.write(f"测试统计:\n")
            f.write(f"  总模型数: {total_models}\n")
            f.write(f"  成功: {successful_models}\n")
            f.write(f"  失败: {failed_models}\n")
            f.write(f"  成功率: {successful_models/total_models*100:.1f}%\n\n")
            
            # 详细结果
            f.write("详细结果:\n")
            f.write("-" * 60 + "\n")
            
            for model_name, result in results.items():
                f.write(f"\n模型: {model_name}\n")
                f.write(f"  类型: {result.get('model_type', 'N/A')}\n")
                f.write(f"  状态: {result.get('status', 'N/A')}\n")
                
                if result.get('status') == 'success':
                    f.write(f"  参数量: {result.get('parameters', 0):,}\n")
                    f.write(f"  训练时间: {result.get('training_time', 0):.2f}s\n")
                    f.write(f"  测试MSE: {result.get('test_mse', 0):.6f}\n")
                    f.write(f"  测试MAE: {result.get('test_mae', 0):.6f}\n")
                    f.write(f"  测试RMSE: {result.get('test_rmse', 0):.6f}\n")
                    f.write(f"  测试R²: {result.get('test_r2', 0):.4f}\n")
                    f.write(f"  推理时间: {result.get('inference_time', 0):.4f}s\n")
                else:
                    f.write(f"  错误: {result.get('error', 'N/A')}\n")
        
        logger.info(f"📊 测试报告已保存: {report_path}")
        return report_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模型测试工具")
    parser.add_argument(
        "--config",
        default="x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/modify_multi_attention/configs/multi_model_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--models",
        help="指定要测试的模型，用逗号分隔（如：transformer,mlp）"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建测试器
        tester = MultiModelTester(args.config)
        
        # 如果指定了特定模型，只测试这些模型
        if args.models:
            selected_models = [m.strip() for m in args.models.split(',')]
            original_models = tester.config.get('models', {})
            filtered_models = {k: v for k, v in original_models.items() if k in selected_models}
            tester.config['models'] = filtered_models
            logger.info(f"🎯 只测试指定模型: {selected_models}")
        
        # 运行测试
        results = tester.run_all_tests()
        
        # 生成报告
        report_path = tester.generate_report(results)
        
        logger.info("\n🎉 测试完成！")
        logger.info(f"📊 报告文件: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {e}")
        logger.error(f"详细错误: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()