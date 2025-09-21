import os
# 设置无头模式环境变量（必须在任何GUI相关导入之前）
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('DISPLAY', '')
os.environ.setdefault('HEADLESS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文显示
# 统一使用全局 sitecustomize.py 的中文字体和负号设置
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

def plot_comparison_figure(input_pressure, true_pressure, predicted_pressure, time_step, epoch, attention_type, idx, parent_dir="attention_results", mode="test"):
    # 创建对应注意力机制的子文件夹
    result_dir = os.path.join(parent_dir, attention_type, "visualization_results")
    os.makedirs(result_dir, exist_ok=True)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(input_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Input Pressure Matrix at t={time_step:.2f}")

    plt.subplot(1, 3, 2)
    plt.imshow(true_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"True Pressure Matrix at t={time_step:.2f}")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_pressure, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Predicted Pressure Matrix at t={time_step:.2f}")

    plt.tight_layout()

    # 仅保存SVG矢量格式，减少磁盘占用
    svg_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}.svg")
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_losses(train_loss, valid_loss, test_loss, save_path=None):
    import matplotlib.pyplot as plt
    if len(train_loss) == 0 or len(valid_loss) == 0 or len(test_loss) == 0:
        print("Warning: 损失列表为空，无法绘制Loss曲线！")
        return

    # 避免对数坐标出现非正值报错，做轻微修正
    eps = 1e-12
    train_vals = [max(float(x), eps) for x in train_loss]
    valid_vals = [max(float(x), eps) for x in valid_loss]
    test_vals = [max(float(x), eps) for x in test_loss]

    plt.figure(figsize=(10, 6))
    plt.plot(train_vals, label='Train Loss')
    plt.plot(valid_vals, label='Valid Loss')
    plt.plot(test_vals, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')  # 使用对数y轴
    plt.title('Loss Curve (Log Scale)')
    plt.legend()

    if save_path:
        # 将任何传入的路径转换为SVG后缀
        import os
        svg_path = os.path.splitext(save_path)[0] + ".svg"
        plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
        print(f"Loss曲线已保存到: {svg_path}")
    else:
        plt.show()

    plt.close()


def plot_difference_figure(true_pressure, predicted_pressure, time_step, epoch, attention_type, idx, parent_dir="attention_results", mode="test"):
    # 创建对应注意力机制的子文件夹
    result_dir = os.path.join(parent_dir, attention_type, "difference_results")
    os.makedirs(result_dir, exist_ok=True)

    # 计算差异（绝对误差）
    difference = np.abs(true_pressure - predicted_pressure)

    plt.figure(figsize=(6, 5))
    plt.imshow(difference, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Difference (|True - Predicted|) at t={time_step:.2f}")

    plt.tight_layout()

    # 仅保存SVG矢量格式，减少磁盘占用
    svg_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}_difference.svg")
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()


def create_model_comparison_plots(results, output_dir="visualization_results", timestamp=None):
    """
    创建模型比较的可视化图表
    
    Args:
        results: 模型测试结果字典
        output_dir: 输出目录
        timestamp: 时间戳字符串
    
    Returns:
        生成的图表文件路径列表
    """
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤成功的结果
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("Warning: 没有成功的模型结果可用于可视化")
        return []
    
    plot_files = []
    
    # 1. 性能对比图 (R² 和 MSE)
    performance_plot = os.path.join(output_dir, f"model_performance_comparison_{timestamp}.png")
    plot_files.append(performance_plot)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = list(successful_results.keys())
    r2_scores = [successful_results[model]['test_r2'] for model in models]
    mse_scores = [successful_results[model]['test_mse'] for model in models]
    
    # R² 分数
    bars1 = ax1.bar(models, r2_scores, color='skyblue', alpha=0.7)
    ax1.set_title('模型R²分数对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom')
    
    # MSE 分数
    bars2 = ax2.bar(models, mse_scores, color='lightcoral', alpha=0.7)
    ax2.set_title('模型MSE对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MSE')
    ax2.set_yscale('log')  # 使用对数坐标
    ax2.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, score in zip(bars2, mse_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 参数量对比图
    params_plot = os.path.join(output_dir, f"model_parameters_comparison_{timestamp}.png")
    plot_files.append(params_plot)
    
    param_counts = [successful_results[model]['num_parameters'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, param_counts, color='lightgreen', alpha=0.7)
    plt.title('模型参数量对比', fontsize=14, fontweight='bold')
    plt.ylabel('参数数量')
    plt.yscale('log')  # 使用对数坐标
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(params_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 训练时间对比图
    time_plot = os.path.join(output_dir, f"model_training_time_comparison_{timestamp}.png")
    plot_files.append(time_plot)
    
    train_times = [successful_results[model]['train_time'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, train_times, color='orange', alpha=0.7)
    plt.title('模型训练时间对比', fontsize=14, fontweight='bold')
    plt.ylabel('训练时间 (秒)')
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值标签
    for bar, time_val in zip(bars, train_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(time_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_files


def generate_model_comparison_summary(results, plot_files, output_dir="visualization_results", timestamp=None):
    """
    生成模型比较的汇总报告
    
    Args:
        results: 模型测试结果
        plot_files: 生成的图表文件列表
        output_dir: 输出目录
        timestamp: 时间戳
    
    Returns:
        汇总报告文件路径
    """
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_path = os.path.join(output_dir, f"model_comparison_summary_{timestamp}.md")
    
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 裁剪模型测试可视化汇总报告\n\n")
        
        from datetime import datetime
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"**测试模型数量**: {len(results)}\n")
        f.write(f"**成功模型数量**: {len(successful_results)}\n")
        f.write(f"**生成图表数量**: {len(plot_files)}\n\n")
        
        f.write("## 📊 生成的可视化图表\n\n")
        for i, plot_file in enumerate(plot_files, 1):
            plot_name = os.path.basename(plot_file)
            f.write(f"{i}. `{plot_name}`\n")
        f.write("\n")
        
        f.write("## 🏆 模型性能汇总\n\n")
        f.write("| 模型 | 测试MSE | 测试R² | 参数量 | 训练时间(s) |\n")
        f.write("|------|---------|--------|--------|-------------|\n")
        
        for model_name, result in successful_results.items():
            f.write(f"| {model_name} | {result['test_mse']:.6f} | "
                   f"{result['test_r2']:.4f} | {result['num_parameters']:,} | "
                   f"{result['train_time']:.2f} |\n")
        
        # 按性能排名
        f.write("\n### 🥇 按性能排名 (MSE越小越好)\n\n")
        sorted_by_mse = sorted(successful_results.items(), key=lambda x: x[1]['test_mse'])
        for i, (model_name, result) in enumerate(sorted_by_mse, 1):
            f.write(f"{i}. **{model_name}**: MSE={result['test_mse']:.6f}\n")
        
        f.write("\n### 🥇 按性能排名 (R²越大越好)\n\n")
        sorted_by_r2 = sorted(successful_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        for i, (model_name, result) in enumerate(sorted_by_r2, 1):
            f.write(f"{i}. **{model_name}**: R²={result['test_r2']:.4f}\n")
    
    return summary_path


def plot_training_losses(train_losses, valid_losses, test_losses, model_name, output_dir="visualization_results", timestamp=None):
    """
    绘制训练过程中的loss曲线
    
    Args:
        train_losses: 训练损失列表
        valid_losses: 验证损失列表  
        test_losses: 测试损失列表
        model_name: 模型名称
        output_dir: 输出目录
        timestamp: 时间戳
    
    Returns:
        loss图表文件路径
    """
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    loss_plot = os.path.join(output_dir, f"{model_name}_loss_curves_{timestamp}.png")
    
    if not train_losses or not valid_losses or not test_losses:
        print(f"Warning: {model_name} 的损失列表为空，无法绘制Loss曲线！")
        return None
    
    # 避免对数坐标出现非正值报错
    eps = 1e-12
    train_vals = [max(float(x), eps) for x in train_losses]
    valid_vals = [max(float(x), eps) for x in valid_losses]
    test_vals = [max(float(x), eps) for x in test_losses]
    
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_vals) + 1)
    
    plt.plot(epochs, train_vals, label='训练损失', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, valid_vals, label='验证损失', linewidth=2, marker='s', markersize=4)
    plt.plot(epochs, test_vals, label='测试损失', linewidth=2, marker='^', markersize=4)
    
    plt.xlabel('训练轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.yscale('log')
    plt.title(f'{model_name} 模型训练损失曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加最终损失值标注
    final_train = train_vals[-1]
    final_valid = valid_vals[-1]
    final_test = test_vals[-1]
    
    plt.text(0.02, 0.98, f'最终训练损失: {final_train:.6f}\n最终验证损失: {final_valid:.6f}\n最终测试损失: {final_test:.6f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {model_name} 损失曲线已保存: {loss_plot}")
    return loss_plot
