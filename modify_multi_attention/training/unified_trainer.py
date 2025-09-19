"""统一训练接口，支持FNO、UNet、PINN、Transformer四种模型的对比实验"""

import torch
import torch.nn as nn
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from timeit import default_timer
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入适配器和模型
from ..data.pdebench_adapter import (
    create_adapted_model, 
    PDEBenchDatasetAdapter
)

# 导入Transformer模型
from ..mymodels.transformer import TransformerFlowReconstructionModel

# 导入训练工具
from .trainer import train_model, test_model
from ..utils.enhanced_svd_loss import create_enhanced_svd_loss
from ..utils.svd10_loss import TotalLossWithSVD
from ..utils.visualization import plot_losses

class UnifiedModelTrainer:
    """统一模型训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化结果存储
        self.results = {}
        
    def create_model(self, model_type: str) -> nn.Module:
        """根据类型创建模型"""
        model_config = self.config['models'][model_type]
        
        if model_type == 'transformer':
            model = TransformerFlowReconstructionModel(
                input_dim=model_config['input_dim'],
                output_dim=model_config['output_dim'],
                num_heads=model_config['num_heads'],
                num_layers=model_config['num_layers'],
                d_model=model_config['d_model'],
                max_time_steps=model_config.get('max_time_steps', 1),
                attention_type=model_config.get('attention_type', 'sge'),
                seq_len=model_config.get('seq_len', 32),
                input_hw=tuple(model_config.get('input_hw')) if model_config.get('input_hw') else None,
                pe_type=model_config.get('pe_type', 'learnable_1d'),
                output_head_type=model_config.get('output_head_type', 'global'),
                out_channels_per_token=model_config.get('out_channels_per_token')
            )
        else:
            # PDEBench模型
            input_resolution = tuple(self.config['data']['input_resolution'])
            output_resolution = tuple(self.config['data']['output_resolution'])
            
            model = create_adapted_model(
                model_type=model_type,
                model_config=model_config,
                input_resolution=input_resolution,
                output_resolution=output_resolution
            )
            
        return model.to(self.device)
    
    def create_dataset(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """创建数据集"""
        data_config = self.config['data']
        
        # 使用适配器创建数据集
        dataset_adapter = PDEBenchDatasetAdapter(
            data_path=data_config['data_path'],
            input_resolution=tuple(data_config['input_resolution']),
            output_resolution=tuple(data_config['output_resolution']),
            num_samples=data_config.get('num_samples', 100),
            normalize_data=data_config.get('normalize_data', True)
        )
        
        # 创建数据加载器
        total_size = len(dataset_adapter)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset_adapter.dataset, [train_size, val_size, test_size]
        )
        
        batch_size = data_config.get('batch_size', 32)
        num_workers = data_config.get('num_workers', 0)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader, dataset_adapter
    
    def create_loss_function(self, model_type: str) -> nn.Module:
        """创建损失函数"""
        loss_config = self.config['training']['loss']
        
        if model_type == 'transformer' and loss_config.get('use_svd_loss', False):
            # Transformer使用SVD损失
            return create_enhanced_svd_loss(
                base_weight=loss_config.get('base_weight', 0.8),
                svd_weights=loss_config.get('svd_weights', [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]),
                topk=loss_config.get('topk', 10),
                mixed_precision=loss_config.get('mixed_precision', False),
                adaptive_weights=loss_config.get('adaptive_weights', True)
            )
        else:
            # 其他模型使用MSE损失
            return nn.MSELoss()
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """创建优化器"""
        optim_config = self.config['training']['optimizer']
        
        if optim_config['type'].lower() == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=optim_config['learning_rate'],
                weight_decay=optim_config.get('weight_decay', 0.0)
            )
        elif optim_config['type'].lower() == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=optim_config['learning_rate'],
                weight_decay=optim_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_config['type']}")
    
    def train_single_model(self, model_type: str, results_dir: Path) -> Dict[str, Any]:
        """训练单个模型"""
        logger.info(f"开始训练 {model_type} 模型")
        
        # 创建模型
        model = self.create_model(model_type)
        logger.info(f"{model_type} 模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 创建数据集
        train_loader, val_loader, test_loader, dataset = self.create_dataset()
        
        # 创建损失函数和优化器
        criterion = self.create_loss_function(model_type)
        optimizer = self.create_optimizer(model)
        
        # 训练配置
        train_config = self.config['training']
        
        # 开始训练
        start_time = default_timer()
        
        if model_type == 'transformer':
            # 使用现有的trainer
            model, train_losses, valid_losses, test_losses = train_model(
                model=model,
                train_loader=train_loader,
                valid_loader=val_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=train_config['epochs'],
                device=self.device,
                early_stop_patience=train_config.get('patience', 15),
                attention_type=self.config['models']['transformer'].get('attention_type', 'sge'),
                result_dir=str(results_dir / model_type),
                cfg=self.config
            )
        else:
            # 简化的训练循环用于PDEBench模型
            train_losses, valid_losses, test_losses = self._train_pdebench_model(
                model, train_loader, val_loader, test_loader, 
                criterion, optimizer, train_config, results_dir / model_type
            )
        
        training_time = default_timer() - start_time
        
        # 最终测试
        final_test_loss = self._evaluate_model(model, test_loader, criterion)
        
        # 保存结果
        result = {
            'model_type': model_type,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'test_losses': test_losses,
            'final_test_loss': final_test_loss,
            'training_time': training_time,
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        
        # 保存模型
        model_path = results_dir / model_type / f'{model_type}_model.pth'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        # 绘制损失曲线
        self._plot_losses(result, results_dir / model_type)
        
        logger.info(f"{model_type} 训练完成，最终测试损失: {final_test_loss:.6f}，训练时间: {training_time:.2f}s")
        
        return result
    
    def _train_pdebench_model(self, model, train_loader, val_loader, test_loader, 
                             criterion, optimizer, train_config, save_dir) -> Tuple[List, List, List]:
        """PDEBench模型的简化训练循环"""
        epochs = train_config['epochs']
        patience = train_config.get('patience', 15)
        
        train_losses = []
        valid_losses = []
        test_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证
            val_loss = self._evaluate_model(model, val_loader, criterion)
            valid_losses.append(val_loss)
            
            # 测试
            test_loss = self._evaluate_model(model, test_loader, criterion)
            test_losses.append(test_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), save_dir / 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"早停在第 {epoch+1} 轮")
                break
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        return train_losses, valid_losses, test_losses
    
    def _evaluate_model(self, model, dataloader, criterion) -> float:
        """评估模型"""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _plot_losses(self, result: Dict[str, Any], save_dir: Path):
        """绘制损失曲线"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(result['train_losses'], label='Train')
        plt.title(f"{result['model_type']} - Training Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(result['valid_losses'], label='Validation')
        plt.title(f"{result['model_type']} - Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(result['test_losses'], label='Test')
        plt.title(f"{result['model_type']} - Test Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / f"{result['model_type']}_losses.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comparison(self, results_dir: Optional[Path] = None) -> Dict[str, Any]:
        """运行模型对比实验"""
        if results_dir is None:
            results_dir = Path('./comparison_results')
        
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取要训练的模型列表
        models_to_train = self.config.get('models_to_train', ['fno2d', 'unet2d', 'transformer'])
        
        logger.info(f"开始对比实验，将训练以下模型: {models_to_train}")
        
        # 训练所有模型
        all_results = {}
        for model_type in models_to_train:
            try:
                result = self.train_single_model(model_type, results_dir)
                all_results[model_type] = result
            except Exception as e:
                logger.error(f"训练 {model_type} 时出错: {e}")
                continue
        
        # 生成对比报告
        self._generate_comparison_report(all_results, results_dir)
        
        # 保存配置
        with open(results_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, sort_keys=False, allow_unicode=True)
        
        logger.info(f"对比实验完成，结果保存在: {results_dir}")
        
        return all_results
    
    def _generate_comparison_report(self, results: Dict[str, Any], save_dir: Path):
        """生成对比报告"""
        # 创建对比表格
        comparison_data = []
        for model_type, result in results.items():
            comparison_data.append({
                'Model': model_type,
                'Final Test Loss': f"{result['final_test_loss']:.6f}",
                'Training Time (s)': f"{result['training_time']:.2f}",
                'Parameters': f"{result['model_parameters']:,}",
                'Best Epoch': len(result['valid_losses'])
            })
        
        # 保存为CSV
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        df.to_csv(save_dir / 'comparison_results.csv', index=False)
        
        # 绘制对比图
        self._plot_comparison(results, save_dir)
        
        # 生成文本报告
        with open(save_dir / 'comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write("模型对比实验报告\n")
            f.write("=" * 50 + "\n\n")
            
            for model_type, result in results.items():
                f.write(f"模型: {model_type}\n")
                f.write(f"  最终测试损失: {result['final_test_loss']:.6f}\n")
                f.write(f"  训练时间: {result['training_time']:.2f}秒\n")
                f.write(f"  参数数量: {result['model_parameters']:,}\n")
                f.write(f"  训练轮数: {len(result['valid_losses'])}\n")
                f.write("\n")
    
    def _plot_comparison(self, results: Dict[str, Any], save_dir: Path):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 最终测试损失对比
        models = list(results.keys())
        test_losses = [results[model]['final_test_loss'] for model in models]
        
        axes[0, 0].bar(models, test_losses)
        axes[0, 0].set_title('Final Test Loss Comparison')
        axes[0, 0].set_ylabel('Test Loss')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 训练时间对比
        training_times = [results[model]['training_time'] for model in models]
        
        axes[0, 1].bar(models, training_times)
        axes[0, 1].set_title('Training Time Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 参数数量对比
        param_counts = [results[model]['model_parameters'] for model in models]
        
        axes[1, 0].bar(models, param_counts)
        axes[1, 0].set_title('Model Parameters Comparison')
        axes[1, 0].set_ylabel('Number of Parameters')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 训练损失曲线对比
        for model in models:
            axes[1, 1].plot(results[model]['train_losses'], label=f'{model} (train)')
            axes[1, 1].plot(results[model]['valid_losses'], label=f'{model} (val)', linestyle='--')
        
        axes[1, 1].set_title('Training Progress Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

# 示例配置
DEFAULT_CONFIG = {
    'data': {
        'data_path': 'path/to/data',
        'input_resolution': [32, 32],
        'output_resolution': [128, 128],
        'num_samples': 1000,
        'normalize_data': True,
        'batch_size': 32,
        'num_workers': 0
    },
    'models': {
        'fno2d': {
            'num_channels': 1,
            'modes1': 12,
            'modes2': 12,
            'width': 20,
            'initial_step': 10
        },
        'unet2d': {
            'in_channels': 1,
            'out_channels': 1,
            'init_features': 32
        },
        'transformer': {
            'input_dim': 1024,
            'output_dim': 16384,
            'num_heads': 8,
            'num_layers': 6,
            'd_model': 512,
            'attention_type': 'sge',
            'seq_len': 1024,
            'input_hw': [32, 32]
        }
    },
    'training': {
        'epochs': 100,
        'patience': 15,
        'optimizer': {
            'type': 'adam',
            'learning_rate': 1e-3,
            'weight_decay': 0.0
        },
        'loss': {
            'use_svd_loss': True,
            'base_weight': 0.8,
            'svd_weights': [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
            'topk': 10
        }
    },
    'models_to_train': ['fno2d', 'unet2d', 'transformer']
}

if __name__ == "__main__":
    # 示例使用
    trainer = UnifiedModelTrainer(DEFAULT_CONFIG)
    results = trainer.run_comparison()
    print("对比实验完成！")