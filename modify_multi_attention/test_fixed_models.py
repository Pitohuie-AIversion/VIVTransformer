"""
测试修复后的MLP和U-Net模型
验证维度问题是否已解决
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
import matplotlib.pyplot as plt

# 导入修复后的模型
from models.fixed_enhanced_mlp import create_fixed_enhanced_mlp1d
from models.fixed_enhanced_unet import create_fixed_enhanced_unet2d

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

class FixedModelTester:
    """修复模型测试器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 测试配置
        self.config = {
            'training': {
                'epochs': 20,
                'batch_size': 16,
                'learning_rate': 1e-3,
                'weight_decay': 1e-5
            },
            'data': {
                'input_dim': 1024,
                'output_dim': 16384,
                'train_samples': 400,
                'val_samples': 100
            }
        }
        
        # 测试结果存储
        self.results = {}
        
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
    
    def test_model_creation(self):
        """测试模型创建"""
        print("="*60)
        print("测试模型创建...")
        print("="*60)
        
        models = {}
        
        # 测试修复版MLP模型
        try:
            print("创建修复版MLP模型...")
            models['mlp'] = create_fixed_enhanced_mlp1d(
                input_channels=1,
                output_channels=1,
                hidden_dim=256,
                num_layers=4,
                input_resolution=32,
                output_resolution=128
            ).to(self.device)
            print(f"✅ MLP模型创建成功！参数数量: {sum(p.numel() for p in models['mlp'].parameters()):,}")
            
        except Exception as e:
            print(f"❌ MLP模型创建失败: {e}")
        
        # 测试修复版U-Net模型
        try:
            print("创建修复版U-Net模型...")
            models['unet'] = create_fixed_enhanced_unet2d(
                in_channels=1,
                out_channels=1,
                init_features=32,
                input_resolution=32,
                output_resolution=128,
                bilinear=False
            ).to(self.device)
            print(f"✅ U-Net模型创建成功！参数数量: {sum(p.numel() for p in models['unet'].parameters()):,}")
            
        except Exception as e:
            print(f"❌ U-Net模型创建失败: {e}")
        
        return models
    
    def test_forward_pass(self, models):
        """测试前向传播"""
        print("\n" + "="*60)
        print("测试前向传播...")
        print("="*60)
        
        # 创建测试输入
        test_input = torch.randn(4, 1024).to(self.device)
        print(f"测试输入形状: {test_input.shape}")
        
        for model_name, model in models.items():
            try:
                print(f"\n测试 {model_name.upper()} 模型前向传播...")
                model.eval()
                with torch.no_grad():
                    output = model(test_input)
                print(f"✅ {model_name.upper()} 前向传播成功！")
                print(f"   输入形状: {test_input.shape}")
                print(f"   输出形状: {output.shape}")
                print(f"   期望输出形状: [4, 16384]")
                
                # 验证输出形状
                if output.shape == (4, 16384):
                    print(f"   ✅ 输出形状正确！")
                else:
                    print(f"   ❌ 输出形状不正确！")
                    
            except Exception as e:
                print(f"❌ {model_name.upper()} 前向传播失败: {e}")
    
    def train_model(self, model, model_name, train_loader, val_loader):
        """训练单个模型"""
        print(f"\n开始训练 {model_name.upper()} 模型...")
        
        train_config = self.config['training']
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # 训练历史
        train_losses = []
        val_losses = []
        
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
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{train_config['epochs']}: "
                          f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
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
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_test_mse': test_mse,
                'final_test_r2': test_r2,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            print(f"✅ {model_name.upper()} 训练完成！")
            print(f"   参数数量: {result['parameters']:,}")
            print(f"   训练时间: {training_time:.2f}秒")
            print(f"   最终训练损失: {train_losses[-1]:.6f}")
            print(f"   最终验证损失: {val_losses[-1]:.6f}")
            print(f"   测试MSE: {test_mse:.6f}")
            print(f"   测试R²: {test_r2:.6f}")
            
            return result
            
        except Exception as e:
            print(f"❌ {model_name.upper()} 训练失败: {e}")
            return {
                'model_name': model_name,
                'status': 'failed',
                'error': str(e),
                'parameters': sum(p.numel() for p in model.parameters()) if model else 0
            }
    
    def run_full_test(self):
        """运行完整测试"""
        print("开始修复模型完整测试...")
        
        # 1. 测试模型创建
        models = self.test_model_creation()
        
        if not models:
            print("❌ 没有成功创建的模型，测试结束")
            return
        
        # 2. 测试前向传播
        self.test_forward_pass(models)
        
        # 3. 创建数据加载器
        print(f"\n" + "="*60)
        print("创建数据加载器...")
        print("="*60)
        train_loader, val_loader = self.create_data_loaders()
        print(f"✅ 数据加载器创建完成：训练样本 {len(train_loader.dataset)}, 验证样本 {len(val_loader.dataset)}")
        
        # 4. 训练模型
        print(f"\n" + "="*60)
        print("开始训练测试...")
        print("="*60)
        
        for model_name, model in models.items():
            result = self.train_model(model, model_name, train_loader, val_loader)
            self.results[model_name] = result
        
        # 5. 保存结果
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """保存测试结果"""
        # 保存详细结果到JSON
        results_file = "fixed_models_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 详细结果已保存到: {results_file}")
        
        # 保存简化结果到文本文件
        summary_file = "fixed_models_test_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("修复模型测试结果摘要\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"模型: {model_name.upper()}\n")
                f.write(f"状态: {result['status']}\n")
                
                if result['status'] == 'success':
                    f.write(f"参数数量: {result['parameters']:,}\n")
                    f.write(f"训练时间: {result['training_time']:.2f}秒\n")
                    f.write(f"最终训练损失: {result['final_train_loss']:.6f}\n")
                    f.write(f"最终验证损失: {result['final_val_loss']:.6f}\n")
                    f.write(f"测试MSE: {result['final_test_mse']:.6f}\n")
                    f.write(f"测试R²: {result['final_test_r2']:.6f}\n")
                else:
                    f.write(f"错误信息: {result.get('error', 'Unknown error')}\n")
                
                f.write("-" * 30 + "\n\n")
        
        print(f"✅ 结果摘要已保存到: {summary_file}")

def main():
    """主函数"""
    tester = FixedModelTester()
    results = tester.run_full_test()
    
    print("\n" + "="*60)
    print("修复模型测试完成！")
    print("="*60)
    
    # 统计成功和失败的模型
    successful = [name for name, result in results.items() if result['status'] == 'success']
    failed = [name for name, result in results.items() if result['status'] == 'failed']
    
    print(f"✅ 成功测试的模型 ({len(successful)}): {', '.join(successful)}")
    if failed:
        print(f"❌ 测试失败的模型 ({len(failed)}): {', '.join(failed)}")
    
    # 显示最佳模型
    if successful:
        if len(successful) > 1:
            best_mse_model = min(successful, key=lambda x: results[x]['final_test_mse'])
            best_r2_model = max(successful, key=lambda x: results[x]['final_test_r2'])
            
            print(f"\n🏆 最佳MSE模型: {best_mse_model.upper()} (MSE: {results[best_mse_model]['final_test_mse']:.6f})")
            print(f"🏆 最佳R²模型: {best_r2_model.upper()} (R²: {results[best_r2_model]['final_test_r2']:.6f})")
        else:
            model_name = successful[0]
            print(f"\n🏆 唯一成功的模型: {model_name.upper()}")
            print(f"   MSE: {results[model_name]['final_test_mse']:.6f}")
            print(f"   R²: {results[model_name]['final_test_r2']:.6f}")

if __name__ == "__main__":
    main()