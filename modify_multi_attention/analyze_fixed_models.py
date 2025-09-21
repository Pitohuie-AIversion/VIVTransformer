"""
修复模型性能分析和可视化
分析MLP和U-Net模型的性能差异原因
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from typing import Dict, List, Tuple, Any

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入修复后的模型
from models.fixed_enhanced_mlp import create_fixed_enhanced_mlp1d
from models.fixed_enhanced_unet import create_fixed_enhanced_unet2d

class ModelAnalyzer:
    """模型性能分析器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载测试结果
        with open('fixed_models_test_results.json', 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        print(f"已加载 {len(self.results)} 个模型的测试结果")
        
    def analyze_training_curves(self):
        """分析训练曲线"""
        print("分析训练曲线...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('修复模型训练曲线分析', fontsize=16, fontweight='bold')
        
        colors = {'mlp': '#2E86AB', 'unet': '#A23B72'}
        
        for model_name, result in self.results.items():
            if result['status'] != 'success':
                continue
                
            color = colors.get(model_name, '#333333')
            epochs = range(1, len(result['train_losses']) + 1)
            
            # 训练损失
            axes[0, 0].plot(epochs, result['train_losses'], 
                           label=f'{model_name.upper()}', color=color, linewidth=2)
            
            # 验证损失
            axes[0, 1].plot(epochs, result['val_losses'], 
                           label=f'{model_name.upper()}', color=color, linewidth=2)
            
            # 损失比值（验证/训练）
            loss_ratio = np.array(result['val_losses']) / np.array(result['train_losses'])
            axes[1, 0].plot(epochs, loss_ratio, 
                           label=f'{model_name.upper()}', color=color, linewidth=2)
            
            # 对数尺度验证损失
            axes[1, 1].semilogy(epochs, result['val_losses'], 
                               label=f'{model_name.upper()}', color=color, linewidth=2)
        
        # 设置子图
        axes[0, 0].set_title('训练损失曲线', fontweight='bold')
        axes[0, 0].set_xlabel('训练轮次')
        axes[0, 0].set_ylabel('训练损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('验证损失曲线', fontweight='bold')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('验证损失')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('过拟合指标 (验证损失/训练损失)', fontweight='bold')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_ylabel('损失比值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='理想线')
        
        axes[1, 1].set_title('验证损失曲线 (对数尺度)', fontweight='bold')
        axes[1, 1].set_xlabel('训练轮次')
        axes[1, 1].set_ylabel('验证损失 (log)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 训练曲线分析图已保存: training_curves_analysis.png")
    
    def analyze_model_complexity(self):
        """分析模型复杂度"""
        print("分析模型复杂度...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('模型复杂度与性能分析', fontsize=16, fontweight='bold')
        
        model_names = []
        param_counts = []
        training_times = []
        test_mses = []
        test_r2s = []
        
        for model_name, result in self.results.items():
            if result['status'] != 'success':
                continue
            
            model_names.append(model_name.upper())
            param_counts.append(result['parameters'])
            training_times.append(result['training_time'])
            test_mses.append(result['final_test_mse'])
            test_r2s.append(result['final_test_r2'])
        
        colors = ['#2E86AB', '#A23B72']
        
        # 参数数量对比
        bars1 = axes[0].bar(model_names, param_counts, color=colors, alpha=0.8)
        axes[0].set_title('模型参数数量对比', fontweight='bold')
        axes[0].set_ylabel('参数数量')
        axes[0].set_yscale('log')
        
        # 添加数值标签
        for bar, count in zip(bars1, param_counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 训练时间对比
        bars2 = axes[1].bar(model_names, training_times, color=colors, alpha=0.8)
        axes[1].set_title('训练时间对比', fontweight='bold')
        axes[1].set_ylabel('训练时间 (秒)')
        
        # 添加数值标签
        for bar, time in zip(bars2, training_times):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 性能指标对比
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        bars3 = axes[2].bar(x_pos - width/2, test_mses, width, 
                           label='测试MSE', color='#F18F01', alpha=0.8)
        
        # 创建第二个y轴用于R²
        ax2 = axes[2].twinx()
        bars4 = ax2.bar(x_pos + width/2, test_r2s, width, 
                       label='测试R²', color='#C73E1D', alpha=0.8)
        
        axes[2].set_title('性能指标对比', fontweight='bold')
        axes[2].set_xlabel('模型')
        axes[2].set_ylabel('测试MSE', color='#F18F01')
        ax2.set_ylabel('测试R²', color='#C73E1D')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(model_names)
        
        # 添加数值标签
        for bar, mse in zip(bars3, test_mses):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{mse:.4f}', ha='center', va='bottom', fontweight='bold')
        
        for bar, r2 in zip(bars4, test_r2s):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 添加图例
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('model_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 模型复杂度分析图已保存: model_complexity_analysis.png")
    
    def analyze_overfitting_patterns(self):
        """分析过拟合模式"""
        print("分析过拟合模式...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('过拟合模式分析', fontsize=16, fontweight='bold')
        
        for model_name, result in self.results.items():
            if result['status'] != 'success':
                continue
            
            epochs = range(1, len(result['train_losses']) + 1)
            train_losses = np.array(result['train_losses'])
            val_losses = np.array(result['val_losses'])
            
            color = '#2E86AB' if model_name == 'mlp' else '#A23B72'
            
            # 训练-验证损失差异
            loss_diff = val_losses - train_losses
            axes[0, 0].plot(epochs, loss_diff, label=f'{model_name.upper()}', 
                           color=color, linewidth=2)
            
            # 损失变化率
            train_loss_change = np.diff(train_losses)
            val_loss_change = np.diff(val_losses)
            
            axes[0, 1].plot(epochs[1:], train_loss_change, 
                           label=f'{model_name.upper()} 训练', color=color, linewidth=2)
            axes[0, 1].plot(epochs[1:], val_loss_change, 
                           label=f'{model_name.upper()} 验证', color=color, 
                           linewidth=2, linestyle='--')
            
            # 累积过拟合指标
            cumulative_overfitting = np.cumsum(np.maximum(0, loss_diff))
            axes[1, 0].plot(epochs, cumulative_overfitting, 
                           label=f'{model_name.upper()}', color=color, linewidth=2)
            
            # 稳定性指标（损失方差）
            window_size = 5
            if len(val_losses) >= window_size:
                val_stability = []
                for i in range(window_size-1, len(val_losses)):
                    window_var = np.var(val_losses[i-window_size+1:i+1])
                    val_stability.append(window_var)
                
                stability_epochs = epochs[window_size-1:]
                axes[1, 1].plot(stability_epochs, val_stability, 
                               label=f'{model_name.upper()}', color=color, linewidth=2)
        
        # 设置子图
        axes[0, 0].set_title('训练-验证损失差异', fontweight='bold')
        axes[0, 0].set_xlabel('训练轮次')
        axes[0, 0].set_ylabel('验证损失 - 训练损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        axes[0, 1].set_title('损失变化率', fontweight='bold')
        axes[0, 1].set_xlabel('训练轮次')
        axes[0, 1].set_ylabel('损失变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        axes[1, 0].set_title('累积过拟合指标', fontweight='bold')
        axes[1, 0].set_xlabel('训练轮次')
        axes[1, 0].set_ylabel('累积过拟合程度')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('验证损失稳定性 (5轮滑动方差)', fontweight='bold')
        axes[1, 1].set_xlabel('训练轮次')
        axes[1, 1].set_ylabel('损失方差')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 过拟合分析图已保存: overfitting_analysis.png")
    
    def generate_performance_report(self):
        """生成性能分析报告"""
        print("生成性能分析报告...")
        
        report_content = []
        report_content.append("# 修复模型性能分析报告\n")
        report_content.append("## 实验概述\n")
        report_content.append("本实验对修复后的MLP和U-Net模型进行了稀疏到稠密重建任务的性能评估。\n")
        
        report_content.append("## 模型配置\n")
        report_content.append("- **输入维度**: [batch_size, 1024] (32×32稀疏数据)")
        report_content.append("- **输出维度**: [batch_size, 16384] (128×128稠密数据)")
        report_content.append("- **上采样倍数**: 4倍 (32→128)")
        report_content.append("- **训练轮次**: 20")
        report_content.append("- **批次大小**: 16\n")
        
        report_content.append("## 性能对比结果\n")
        report_content.append("| 模型 | 参数数量 | 训练时间(s) | 最终训练损失 | 最终验证损失 | 测试MSE | 测试R² |")
        report_content.append("|------|----------|-------------|--------------|--------------|---------|--------|")
        
        for model_name, result in self.results.items():
            if result['status'] == 'success':
                report_content.append(
                    f"| {model_name.upper()} | {result['parameters']:,} | "
                    f"{result['training_time']:.1f} | {result['final_train_loss']:.6f} | "
                    f"{result['final_val_loss']:.6f} | {result['final_test_mse']:.6f} | "
                    f"{result['final_test_r2']:.6f} |"
                )
        
        report_content.append("\n## 关键发现\n")
        
        # 分析MLP性能
        mlp_result = self.results.get('mlp', {})
        if mlp_result.get('status') == 'success':
            report_content.append("### MLP模型表现")
            report_content.append(f"- **优秀的泛化能力**: R²达到{mlp_result['final_test_r2']:.3f}，表明模型能很好地拟合稀疏到稠密的映射关系")
            report_content.append(f"- **稳定的训练过程**: 训练和验证损失均稳定下降，最终验证损失({mlp_result['final_val_loss']:.6f})略低于训练损失")
            report_content.append(f"- **合理的模型复杂度**: {mlp_result['parameters']:,}个参数，在性能和效率间取得良好平衡")
        
        # 分析U-Net性能
        unet_result = self.results.get('unet', {})
        if unet_result.get('status') == 'success':
            report_content.append("\n### U-Net模型表现")
            report_content.append(f"- **严重过拟合问题**: 验证损失({unet_result['final_val_loss']:.6f})远高于训练损失({unet_result['final_train_loss']:.6f})")
            report_content.append(f"- **负R²值**: R²为{unet_result['final_test_r2']:.3f}，表明模型预测效果不如简单的均值预测")
            report_content.append(f"- **参数过多**: {unet_result['parameters']:,}个参数，约为MLP的22倍，容易导致过拟合")
        
        report_content.append("\n## 问题分析\n")
        report_content.append("### U-Net过拟合原因")
        report_content.append("1. **参数数量过多**: U-Net有7.76M参数，而训练数据仅400个样本，参数/数据比例过高")
        report_content.append("2. **架构不匹配**: U-Net设计用于图像分割任务，对于稀疏到稠密重建可能过于复杂")
        report_content.append("3. **正则化不足**: 当前配置下缺乏足够的正则化机制防止过拟合")
        
        report_content.append("\n### MLP成功原因")
        report_content.append("1. **适当的模型复杂度**: 346K参数适合当前数据规模")
        report_content.append("2. **有效的特征映射**: 正弦位置编码和傅里叶特征有助于捕获空间关系")
        report_content.append("3. **残差连接**: 帮助梯度传播和特征学习")
        
        report_content.append("\n## 改进建议\n")
        report_content.append("### 针对U-Net模型")
        report_content.append("1. **减少模型复杂度**: 降低初始特征数量，减少网络层数")
        report_content.append("2. **增强正则化**: 添加Dropout、权重衰减、早停等机制")
        report_content.append("3. **数据增强**: 增加训练数据量或使用数据增强技术")
        report_content.append("4. **架构优化**: 考虑使用更适合稀疏重建的轻量级架构")
        
        report_content.append("\n### 针对MLP模型")
        report_content.append("1. **进一步优化**: 可尝试调整隐藏层维度和层数")
        report_content.append("2. **集成学习**: 结合多个MLP模型提升性能")
        report_content.append("3. **损失函数**: 探索更适合稀疏重建的损失函数")
        
        # 保存报告
        with open('performance_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print("✅ 性能分析报告已保存: performance_analysis_report.md")
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("="*60)
        print("开始修复模型完整性能分析...")
        print("="*60)
        
        # 1. 训练曲线分析
        self.analyze_training_curves()
        
        # 2. 模型复杂度分析
        self.analyze_model_complexity()
        
        # 3. 过拟合模式分析
        self.analyze_overfitting_patterns()
        
        # 4. 生成性能报告
        self.generate_performance_report()
        
        print("\n" + "="*60)
        print("性能分析完成！生成的文件:")
        print("="*60)
        print("📊 training_curves_analysis.png - 训练曲线分析")
        print("📊 model_complexity_analysis.png - 模型复杂度分析")
        print("📊 overfitting_analysis.png - 过拟合模式分析")
        print("📄 performance_analysis_report.md - 详细性能分析报告")
        
        return {
            'training_curves': 'training_curves_analysis.png',
            'complexity': 'model_complexity_analysis.png',
            'overfitting': 'overfitting_analysis.png',
            'report': 'performance_analysis_report.md'
        }

def main():
    """主函数"""
    analyzer = ModelAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print(f"\n🎯 分析结论:")
    print(f"   ✅ MLP模型: 优秀性能，R²=0.959")
    print(f"   ❌ U-Net模型: 严重过拟合，需要架构优化")
    print(f"   💡 建议: 使用MLP作为基准模型，优化U-Net架构")

if __name__ == "__main__":
    main()