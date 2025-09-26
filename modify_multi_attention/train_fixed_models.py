"""
修复版模型训练脚本
使用修复后的MLP和U-Net模型进行四模型对比实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Any
import yaml
import matplotlib.pyplot as plt

# 导入修复后的模型
from models.fixed_enhanced_mlp import create_fixed_enhanced_mlp1d
from models.fixed_enhanced_unet import create_fixed_enhanced_unet2d

# 导入原有的正常工作的模型
from models.enhanced_transformer import create_enhanced_transformer1d
from models.enhanced_fno import create_enhanced_fno2d

class SimpleDataset(Dataset):
    """简化的数据集，用于测试模型兼容性"""
    
    def __init__(self, num_samples=1000, input_dim=1024, output_dim=16384, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 生成稀疏输入数据 [num_samples, input_dim]
        self.inputs = torch.randn(num_samples, input_dim) * 0.5
        
        # 生成对应的稠密输出数据 [num_samples, output_dim]
        # 使用简单的上采样和噪声模拟真实的稀疏到稠密映射
        input_2d = self.inputs.view(num_samples, 1, 32, 32)
        output_2d = torch.nn.functional.interpolate(
            input_2d, size=(128, 128), mode='bilinear', align_corners=False
        )
        # 添加一些非线性变换和噪声
        output_2d = torch.sin(output_2d * 2) + torch.randn_like(output_2d) * 0.1
        self.outputs = output_2d.view(num_samples, output_dim)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class FixedModelTrainer:
    """修复版模型训练器"""
    
    def __init__(self, config_path="unified_training_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 训练结果存储
        self.results = {}
        
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"配置文件 {config_path} 未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'weight_decay': 1e-5,
                'patience': 10
            },
            'data': {
                'input_dim': 1024,
                'output_dim': 16384,
                'train_samples': 800,
                'val_samples': 200
            },
            'models': {
                'transformer': {
                    'hidden_dim': 256,
                    'num_layers': 6,
                    'num_heads': 8
                },
                'fno': {
                    'hidden_channels': 64,
                    'num_layers': 4,
                    'modes': 16
                },
                'mlp': {
                    'hidden_dim': 512,
                    'num_layers': 6
                },
                'unet': {
                    'init_features': 64,
                    'bilinear': False
                }
            }
        }
    
    def create_data_loaders(self):
        """创建数据加载器"""
        train_config = self.config['training']
        data_config = self.config['data']
        
        # 创建训练和验证数据集
        train_dataset = SimpleDataset(
            num_samples=data_config['train_samples'],
            input_dim=data_config['input_dim'],
            output_dim=data_config['output_dim'],
            seed=42
        )
        
        val_dataset = SimpleDataset(
            num_samples=data_config['val_samples'],
            input_dim=data_config['input_dim'],
            output_dim=data_config['output_dim'],
            seed=123
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def create_models(self):
        """创建所有模型"""
        models = {}
        model_configs = self.config['models']
        
        try:
            # 1. Transformer模型（已验证工作正常）
            print("创建Transformer模型...")
            models['transformer'] = create_enhanced_transformer1d(
                input_channels=1,
                output_channels=1,
                hidden_dim=model_configs['transformer']['hidden_dim'],
                num_layers=model_configs['transformer']['num_layers'],
                num_heads=model_configs['transformer']['num_heads'],
                input_resolution=32,
                output_resolution=128
            ).to(self.device)
            print(f"Transformer参数数量: {sum(p.numel() for p in models['transformer'].parameters()):,}")
            
        except Exception as e:
            print(f"Transformer模型创建失败: {e}")
        
        try:
            # 2. FNO模型（已验证工作正常）
            print("创建FNO模型...")
            models['fno'] = create_enhanced_fno2d(
                num_channels=1,
                modes1=model_configs['fno']['modes'],
                modes2=model_configs['fno']['modes'],
                width=model_configs['fno']['hidden_channels'],
                input_resolution=(32, 32),
                output_resolution=(128, 128)
            ).to(self.device)
            print(f"FNO参数数量: {sum(p.numel() for p in models['fno'].parameters()):,}")
            
        except Exception as e:
            print(f"FNO模型创建失败: {e}")
        
        try:
            # 3. 修复版MLP模型
            print("创建修复版MLP模型...")
            models['mlp'] = create_fixed_enhanced_mlp1d(
                input_channels=1,
                output_channels=1,
                hidden_dim=model_configs['mlp']['hidden_dim'],
                num_layers=model_configs['mlp']['num_layers'],
                input_resolution=32,
                output_resolution=128
            ).to(self.device)
            print(f"修复版MLP参数数量: {sum(p.numel() for p in models['mlp'].parameters()):,}")
            
        except Exception as e:
            print(f"修复版MLP模型创建失败: {e}")
        
        try:
            # 4. 修复版U-Net模型
            print("创建修复版U-Net模型...")
            models['unet'] = create_fixed_enhanced_unet2d(
                in_channels=1,
                out_channels=1,
                init_features=model_configs['unet']['init_features'],
                input_resolution=32,
                output_resolution=128,
                bilinear=model_configs['unet']['bilinear']
            ).to(self.device)
            print(f"修复版U-Net参数数量: {sum(p.numel() for p in models['unet'].parameters()):,}")
            
        except Exception as e:
            print(f"修复版U-Net模型创建失败: {e}")
        
        return models
    
    def train_model(self, model, model_name, train_loader, val_loader):
        """训练单个模型"""
        print(f"\n开始训练 {model_name} 模型...")
        
        train_config = self.config['training']
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=train_config['patience']//2, factor=0.5
        )
        
        # 训练历史
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        try:
            for epoch in range(train_config['epochs']):
                # 训练阶段
                model.train()
                train_loss = 0.0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                # 验证阶段
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # 学习率调度
                scheduler.step(avg_val_loss)
                
                # 早停检查
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{train_config['epochs']}: "
                          f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                if patience_counter >= train_config['patience']:
                    print(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break
            
            training_time = time.time() - start_time
            
            # 计算最终测试指标
            model.eval()
            test_mse = 0.0
            test_r2_num = 0.0
            test_r2_den = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    
                    # MSE
                    mse = nn.functional.mse_loss(outputs, targets)
                    test_mse += mse.item()
                    
                    # R²计算
                    targets_mean = targets.mean()
                    test_r2_num += ((outputs - targets) ** 2).sum().item()
                    test_r2_den += ((targets - targets_mean) ** 2).sum().item()
            
            test_mse /= len(val_loader)
            test_r2 = 1 - (test_r2_num / test_r2_den) if test_r2_den > 0 else 0.0
            
            # 保存结果
            result = {
                'model_name': model_name,
                'status': 'success',
                'parameters': sum(p.numel() for p in model.parameters()),
                'training_time': training_time,
                'epochs_trained': len(train_losses),
                'best_val_loss': best_val_loss,
                'final_test_mse': test_mse,
                'final_test_r2': test_r2,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            print(f"{model_name} 训练完成！")
            print(f"参数数量: {result['parameters']:,}")
            print(f"训练时间: {training_time:.2f}秒")
            print(f"最佳验证损失: {best_val_loss:.6f}")
            print(f"测试MSE: {test_mse:.6f}")
            print(f"测试R²: {test_r2:.6f}")
            
            return result
            
        except Exception as e:
            print(f"{model_name} 训练失败: {e}")
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e),
                'parameters': sum(p.numel() for p in model.parameters()) if model else 0
            }
    
    def train_all_models(self):
        """训练所有模型"""
        print("开始修复版四模型对比实验...")
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders()
        print(f"数据加载器创建完成：训练样本 {len(train_loader.dataset)}, 验证样本 {len(val_loader.dataset)}")
        
        # 创建模型
        models = self.create_models()
        print(f"成功创建 {len(models)} 个模型")
        
        # 训练每个模型
        for model_name, model in models.items():
            result = self.train_model(model, model_name, train_loader, val_loader)
            self.results[model_name] = result
        
        # 保存结果
        self.save_results()
        self.generate_comparison_report()
        
        return self.results
    
    def save_results(self):
        """保存训练结果"""
        # 保存详细结果到JSON
        results_file = "fixed_models_training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"详细结果已保存到: {results_file}")
        
        # 保存简化结果到文本文件
        summary_file = "fixed_models_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("修复版四模型对比实验结果摘要\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"模型: {model_name.upper()}\n")
                f.write(f"状态: {result['status']}\n")
                
                if result['status'] == 'success':
                    f.write(f"参数数量: {result['parameters']:,}\n")
                    f.write(f"训练时间: {result['training_time']:.2f}秒\n")
                    f.write(f"最佳验证损失: {result['best_val_loss']:.6f}\n")
                    f.write(f"测试MSE: {result['final_test_mse']:.6f}\n")
                    f.write(f"测试R²: {result['final_test_r2']:.6f}\n")
                else:
                    f.write(f"错误信息: {result.get('error', 'Unknown error')}\n")
                
                f.write("-" * 30 + "\n\n")
        
        print(f"结果摘要已保存到: {summary_file}")
    
    def generate_comparison_report(self):
        """生成对比分析报告"""
        successful_models = {k: v for k, v in self.results.items() if v['status'] == 'success'}
        
        if not successful_models:
            print("没有成功训练的模型，无法生成对比报告")
            return
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('修复版四模型对比分析', fontsize=16, fontweight='bold')
        
        # 1. 参数数量对比
        model_names = list(successful_models.keys())
        param_counts = [successful_models[name]['parameters'] for name in model_names]
        
        axes[0, 0].bar(model_names, param_counts, color=['blue', 'green', 'red', 'orange'][:len(model_names)])
        axes[0, 0].set_title('模型参数数量对比')
        axes[0, 0].set_ylabel('参数数量')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 测试MSE对比
        test_mses = [successful_models[name]['final_test_mse'] for name in model_names]
        
        axes[0, 1].bar(model_names, test_mses, color=['blue', 'green', 'red', 'orange'][:len(model_names)])
        axes[0, 1].set_title('测试MSE对比（越低越好）')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 测试R²对比
        test_r2s = [successful_models[name]['final_test_r2'] for name in model_names]
        
        axes[1, 0].bar(model_names, test_r2s, color=['blue', 'green', 'red', 'orange'][:len(model_names)])
        axes[1, 0].set_title('测试R²对比（越高越好）')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 训练时间对比
        training_times = [successful_models[name]['training_time'] for name in model_names]
        
        axes[1, 1].bar(model_names, training_times, color=['blue', 'green', 'red', 'orange'][:len(model_names)])
        axes[1, 1].set_title('训练时间对比（秒）')
        axes[1, 1].set_ylabel('时间（秒）')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('fixed_models_comparison.svg', bbox_inches='tight', format='svg')
        plt.close()
        
        print("对比图表已保存到: fixed_models_comparison.svg")

def main():
    """主函数"""
    trainer = FixedModelTrainer()
    results = trainer.train_all_models()
    
    print("\n" + "="*60)
    print("修复版四模型对比实验完成！")
    print("="*60)
    
    # 统计成功和失败的模型
    successful = [name for name, result in results.items() if result['status'] == 'success']
    failed = [name for name, result in results.items() if result['status'] == 'failed']
    
    print(f"成功训练的模型 ({len(successful)}): {', '.join(successful)}")
    if failed:
        print(f"训练失败的模型 ({len(failed)}): {', '.join(failed)}")
    
    # 显示最佳模型
    if successful:
        best_mse_model = min(successful, key=lambda x: results[x]['final_test_mse'])
        best_r2_model = max(successful, key=lambda x: results[x]['final_test_r2'])
        
        print(f"\n最佳MSE模型: {best_mse_model} (MSE: {results[best_mse_model]['final_test_mse']:.6f})")
        print(f"最佳R²模型: {best_r2_model} (R²: {results[best_r2_model]['final_test_r2']:.6f})")

if __name__ == "__main__":
    main()