#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版归一化测试脚本

功能:
1. 测试不同归一化方法的效果
2. 比较归一化前后的模型性能
3. 可视化归一化效果
4. 生成详细的归一化分析报告

使用方法:
python run_enhanced_normalization_test.py --config configs/enhanced_normalization_config.yaml --methods minmax zscore robust

作者: AI Assistant
日期: 2025
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from data.enhanced_crop_dataloader import create_enhanced_crop_dataloader
from models.mlp_model import MLPModel
from models.transformer import VisionTransformer
from utils.trainer import Trainer
from utils.visualization import plot_training_losses, generate_model_comparison_summary

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NormalizationTester:
    """归一化测试器"""
    
    def __init__(self, config_path):
        """
        初始化归一化测试器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 self.config.get('device', {}).get('use_cuda', True) else 'cpu')
        
        # 创建输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"normalization_test_results_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"归一化测试器初始化完成")
        logger.info(f"使用设备: {self.device}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def test_normalization_methods(self, methods=['minmax', 'zscore', 'robust', 'none']):
        """
        测试不同归一化方法
        
        Args:
            methods: 要测试的归一化方法列表
            
        Returns:
            dict: 测试结果
        """
        results = {}
        
        for method in methods:
            logger.info(f"\n{'='*50}")
            logger.info(f"测试归一化方法: {method}")
            logger.info(f"{'='*50}")
            
            # 修改配置
            test_config = self.config.copy()
            if method == 'none':
                test_config['data']['normalize'] = False
            else:
                test_config['data']['normalize'] = True
                test_config['data']['normalize_method'] = method
                if method == 'minmax':
                    test_config['data']['normalize_range'] = [0, 1]
            
            # 测试该方法
            result = self._test_single_method(method, test_config)
            results[method] = result
            
            # 保存中间结果
            self._save_method_result(method, result)
        
        # 生成比较报告
        self._generate_comparison_report(results)
        
        return results
    
    def _test_single_method(self, method_name, config):
        """
        测试单个归一化方法
        
        Args:
            method_name: 方法名称
            config: 配置字典
            
        Returns:
            dict: 测试结果
        """
        try:
            # 创建数据加载器
            train_loader, val_loader, test_loader, norm_stats = create_enhanced_crop_dataloader(config)
            
            # 获取数据维度
            sample_input, sample_output = next(iter(train_loader))
            input_dim = sample_input.shape[1]
            output_dim = sample_output.shape[1]
            
            logger.info(f"数据维度: 输入={input_dim}, 输出={output_dim}")
            
            # 创建模型
            model = self._create_model(config, input_dim, output_dim)
            model = model.to(self.device)
            
            # 训练模型
            trainer = Trainer(model, self.device)
            train_losses, val_losses = trainer.train(
                train_loader, val_loader,
                epochs=config['training']['epochs'],
                learning_rate=config['training']['learning_rate']
            )
            
            # 评估模型
            test_metrics = trainer.evaluate(test_loader)
            
            # 分析数据分布
            data_analysis = self._analyze_data_distribution(train_loader, norm_stats)
            
            result = {
                'method': method_name,
                'normalization_stats': norm_stats,
                'data_analysis': data_analysis,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_metrics': test_metrics,
                'model_params': sum(p.numel() for p in model.parameters()),
                'input_dim': input_dim,
                'output_dim': output_dim
            }
            
            logger.info(f"方法 {method_name} 测试完成")
            logger.info(f"最终测试MSE: {test_metrics['mse']:.6f}")
            logger.info(f"最终测试R²: {test_metrics['r2']:.6f}")
            
            return result
            
        except Exception as e:
            logger.error(f"测试方法 {method_name} 时出错: {str(e)}")
            return {
                'method': method_name,
                'error': str(e),
                'success': False
            }
    
    def _create_model(self, config, input_dim, output_dim):
        """创建模型"""
        model_config = config['model']
        model_type = model_config['type']
        
        if model_type == 'mlp':
            mlp_config = model_config['mlp']
            model = MLPModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=mlp_config['hidden_dims'],
                activation=mlp_config.get('activation', 'relu'),
                dropout=mlp_config.get('dropout', 0.1),
                batch_norm=mlp_config.get('batch_norm', True)
            )
        elif model_type == 'transformer':
            transformer_config = model_config['transformer']
            model = VisionTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=transformer_config['d_model'],
                nhead=transformer_config['nhead'],
                num_layers=transformer_config['num_layers'],
                dim_feedforward=transformer_config['dim_feedforward'],
                dropout=transformer_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model
    
    def _analyze_data_distribution(self, data_loader, norm_stats):
        """
        分析数据分布
        
        Args:
            data_loader: 数据加载器
            norm_stats: 归一化统计信息
            
        Returns:
            dict: 数据分析结果
        """
        inputs_list = []
        outputs_list = []
        
        # 收集数据样本
        for inputs, outputs in data_loader:
            inputs_list.append(inputs)
            outputs_list.append(outputs)
            if len(inputs_list) >= 10:  # 限制样本数量
                break
        
        all_inputs = torch.cat(inputs_list, dim=0)
        all_outputs = torch.cat(outputs_list, dim=0)
        
        analysis = {
            'input_stats': {
                'mean': all_inputs.mean().item(),
                'std': all_inputs.std().item(),
                'min': all_inputs.min().item(),
                'max': all_inputs.max().item(),
                'median': all_inputs.median().item()
            },
            'output_stats': {
                'mean': all_outputs.mean().item(),
                'std': all_outputs.std().item(),
                'min': all_outputs.min().item(),
                'max': all_outputs.max().item(),
                'median': all_outputs.median().item()
            },
            'normalization_info': norm_stats
        }
        
        return analysis
    
    def _save_method_result(self, method_name, result):
        """保存单个方法的结果"""
        # 保存训练曲线
        if 'train_losses' in result and 'val_losses' in result:
            self._plot_training_curves(method_name, result)
        
        # 保存数据分布图
        if 'data_analysis' in result:
            self._plot_data_distribution(method_name, result['data_analysis'])
    
    def _plot_training_curves(self, method_name, result):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(result['train_losses'], label='训练损失', color='blue')
        plt.plot(result['val_losses'], label='验证损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{method_name} - 训练曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失对数图
        plt.subplot(1, 2, 2)
        plt.semilogy(result['train_losses'], label='训练损失', color='blue')
        plt.semilogy(result['val_losses'], label='验证损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'{method_name} - 训练曲线 (对数)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{method_name}_training_curves.svg', bbox_inches='tight', format='svg')
        plt.close()
    
    def _plot_data_distribution(self, method_name, data_analysis):
        """绘制数据分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        input_stats = data_analysis['input_stats']
        output_stats = data_analysis['output_stats']
        
        # 输入数据统计
        axes[0, 0].bar(['均值', '标准差', '最小值', '最大值', '中位数'],
                      [input_stats['mean'], input_stats['std'], input_stats['min'], 
                       input_stats['max'], input_stats['median']])
        axes[0, 0].set_title(f'{method_name} - 输入数据统计')
        axes[0, 0].set_ylabel('数值')
        
        # 输出数据统计
        axes[0, 1].bar(['均值', '标准差', '最小值', '最大值', '中位数'],
                      [output_stats['mean'], output_stats['std'], output_stats['min'], 
                       output_stats['max'], output_stats['median']])
        axes[0, 1].set_title(f'{method_name} - 输出数据统计')
        axes[0, 1].set_ylabel('数值')
        
        # 归一化信息
        norm_info = data_analysis.get('normalization_info', {})
        if norm_info:
            info_text = f"归一化方法: {norm_info.get('normalize_method', 'N/A')}\n"
            info_text += f"归一化范围: {norm_info.get('normalize_range', 'N/A')}\n"
            
            input_norm = norm_info.get('input_stats', {})
            if input_norm:
                info_text += f"\n输入归一化统计:\n"
                for key, value in input_norm.items():
                    if isinstance(value, (int, float)):
                        info_text += f"  {key}: {value:.6f}\n"
                    else:
                        info_text += f"  {key}: {value}\n"
            
            axes[1, 0].text(0.1, 0.9, info_text, transform=axes[1, 0].transAxes, 
                           verticalalignment='top', fontsize=10, fontfamily='monospace')
            axes[1, 0].set_title(f'{method_name} - 归一化信息')
            axes[1, 0].axis('off')
        
        # 数据范围比较
        ranges = {
            '输入范围': input_stats['max'] - input_stats['min'],
            '输出范围': output_stats['max'] - output_stats['min']
        }
        axes[1, 1].bar(ranges.keys(), ranges.values())
        axes[1, 1].set_title(f'{method_name} - 数据范围')
        axes[1, 1].set_ylabel('范围大小')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{method_name}_data_distribution.svg', bbox_inches='tight', format='svg')
        plt.close()
    
    def _generate_comparison_report(self, results):
        """生成比较报告"""
        report_path = self.output_dir / f'normalization_comparison_report_{self.timestamp}.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 归一化方法比较报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 测试概览
            f.write(f"## 测试概览\n\n")
            f.write(f"- 测试方法数量: {len(results)}\n")
            f.write(f"- 成功方法数量: {sum(1 for r in results.values() if 'error' not in r)}\n")
            f.write(f"- 输出目录: {self.output_dir}\n\n")
            
            # 方法列表
            f.write(f"## 测试方法\n\n")
            for method, result in results.items():
                if 'error' in result:
                    f.write(f"- **{method}**: ❌ 失败 - {result['error']}\n")
                else:
                    f.write(f"- **{method}**: ✅ 成功\n")
            f.write(f"\n")
            
            # 性能比较表
            f.write(f"## 性能比较\n\n")
            f.write(f"| 方法 | 最终MSE | 最终R² | 最终训练损失 | 最终验证损失 | 模型参数 |\n")
            f.write(f"|------|---------|--------|-------------|-------------|----------|\n")
            
            for method, result in results.items():
                if 'error' not in result:
                    mse = result['test_metrics']['mse']
                    r2 = result['test_metrics']['r2']
                    final_train_loss = result['train_losses'][-1] if result['train_losses'] else 'N/A'
                    final_val_loss = result['val_losses'][-1] if result['val_losses'] else 'N/A'
                    params = result['model_params']
                    
                    f.write(f"| {method} | {mse:.6f} | {r2:.6f} | {final_train_loss:.6f} | {final_val_loss:.6f} | {params:,} |\n")
            f.write(f"\n")
            
            # 按性能排序
            successful_results = {k: v for k, v in results.items() if 'error' not in v}
            if successful_results:
                # 按MSE排序
                sorted_by_mse = sorted(successful_results.items(), key=lambda x: x[1]['test_metrics']['mse'])
                f.write(f"## 性能排名\n\n")
                f.write(f"### 按MSE排序 (越小越好)\n\n")
                for i, (method, result) in enumerate(sorted_by_mse, 1):
                    mse = result['test_metrics']['mse']
                    f.write(f"{i}. **{method}**: {mse:.6f}\n")
                f.write(f"\n")
                
                # 按R²排序
                sorted_by_r2 = sorted(successful_results.items(), key=lambda x: x[1]['test_metrics']['r2'], reverse=True)
                f.write(f"### 按R²排序 (越大越好)\n\n")
                for i, (method, result) in enumerate(sorted_by_r2, 1):
                    r2 = result['test_metrics']['r2']
                    f.write(f"{i}. **{method}**: {r2:.6f}\n")
                f.write(f"\n")
            
            # 详细分析
            f.write(f"## 详细分析\n\n")
            for method, result in results.items():
                if 'error' not in result:
                    f.write(f"### {method}\n\n")
                    
                    # 归一化信息
                    norm_stats = result.get('normalization_stats', {})
                    if norm_stats:
                        f.write(f"**归一化配置:**\n")
                        f.write(f"- 方法: {norm_stats.get('normalize_method', 'N/A')}\n")
                        f.write(f"- 范围: {norm_stats.get('normalize_range', 'N/A')}\n\n")
                    
                    # 数据分析
                    data_analysis = result.get('data_analysis', {})
                    if data_analysis:
                        input_stats = data_analysis['input_stats']
                        output_stats = data_analysis['output_stats']
                        
                        f.write(f"**数据统计:**\n")
                        f.write(f"- 输入范围: [{input_stats['min']:.6f}, {input_stats['max']:.6f}]\n")
                        f.write(f"- 输入均值±标准差: {input_stats['mean']:.6f}±{input_stats['std']:.6f}\n")
                        f.write(f"- 输出范围: [{output_stats['min']:.6f}, {output_stats['max']:.6f}]\n")
                        f.write(f"- 输出均值±标准差: {output_stats['mean']:.6f}±{output_stats['std']:.6f}\n\n")
                    
                    # 性能指标
                    test_metrics = result['test_metrics']
                    f.write(f"**性能指标:**\n")
                    for metric, value in test_metrics.items():
                        f.write(f"- {metric.upper()}: {value:.6f}\n")
                    f.write(f"\n")
                    
                    # 生成的图表
                    f.write(f"**生成图表:**\n")
                    f.write(f"- 训练曲线: `{method}_training_curves.png`\n")
                    f.write(f"- 数据分布: `{method}_data_distribution.png`\n\n")
            
            # 使用建议
            f.write(f"## 使用建议\n\n")
            if successful_results:
                best_mse_method = min(successful_results.items(), key=lambda x: x[1]['test_metrics']['mse'])[0]
                best_r2_method = max(successful_results.items(), key=lambda x: x[1]['test_metrics']['r2'])[0]
                
                f.write(f"- **最佳MSE性能**: {best_mse_method}\n")
                f.write(f"- **最佳R²性能**: {best_r2_method}\n\n")
                
                f.write(f"**归一化方法选择建议:**\n")
                f.write(f"1. **Min-Max归一化**: 适用于数据分布相对均匀的情况，将数据缩放到指定范围\n")
                f.write(f"2. **Z-score标准化**: 适用于数据呈正态分布的情况，去除量纲影响\n")
                f.write(f"3. **鲁棒归一化**: 适用于数据包含异常值的情况，使用中位数和四分位距\n")
                f.write(f"4. **不归一化**: 适用于数据已经在合适范围内的情况\n\n")
        
        logger.info(f"比较报告已保存到: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='归一化方法测试')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--methods', nargs='+', default=['minmax', 'zscore', 'robust', 'none'],
                       help='要测试的归一化方法')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = NormalizationTester(args.config)
    
    # 运行测试
    results = tester.test_normalization_methods(args.methods)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"归一化测试完成!")
    logger.info(f"结果保存在: {tester.output_dir}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()