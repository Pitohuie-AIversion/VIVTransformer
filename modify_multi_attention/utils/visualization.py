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

    # 改进的数据处理函数，保持原始分辨率差异
    def ensure_2d_preserve_resolution(data, name):
        """保持原始分辨率的数据处理"""
        if len(data.shape) == 1:
            size = data.shape[0]
            # 根据数据大小确定分辨率
            if size == 16384:  # 128x128
                reshaped = data.reshape(128, 128)
                resolution_info = "128×128"
            elif size == 1024:  # 32x32
                reshaped = data.reshape(32, 32)
                resolution_info = "32×32"
            else:
                side_len = int(np.sqrt(size))
                if side_len * side_len == size:
                    reshaped = data.reshape(side_len, side_len)
                    resolution_info = f"{side_len}×{side_len}"
                else:
                    # 简化的因子分解
                    for i in range(int(np.sqrt(size)), 0, -1):
                        if size % i == 0:
                            reshaped = data.reshape(i, size // i)
                            resolution_info = f"{i}×{size//i}"
                            break
                    else:
                        reshaped = data.reshape(1, -1)
                        resolution_info = f"1×{size}"
        elif len(data.shape) == 2:
            reshaped = data
            resolution_info = f"{data.shape[0]}×{data.shape[1]}"
        else:
            raise ValueError(f"{name} 数据维度不支持: {data.shape}")
        
        return reshaped, resolution_info
    
    try:
        # 使用改进的处理函数，保持原始分辨率
        input_2d, input_res = ensure_2d_preserve_resolution(input_pressure, "input_pressure")
        true_2d, true_res = ensure_2d_preserve_resolution(true_pressure, "true_pressure")
        pred_2d, pred_res = ensure_2d_preserve_resolution(predicted_pressure, "predicted_pressure")
        
        # 打印分辨率信息用于调试
        print(f"分辨率信息 - 输入: {input_res}, 真实: {true_res}, 预测: {pred_res}")
        
    except Exception as e:
        print(f"数据维度处理失败: {e}")
        return

    # 创建图形，使用不同的子图尺寸来反映分辨率差异
    fig = plt.figure(figsize=(18, 6))
    
    # 根据分辨率调整子图布局
    input_h, input_w = input_2d.shape
    true_h, true_w = true_2d.shape
    pred_h, pred_w = pred_2d.shape
    
    # 计算统一的colorbar范围
    try:
        # 计算所有数据的全局最小值和最大值，确保colorbar范围一致
        global_min = min(np.min(input_2d), np.min(true_2d), np.min(pred_2d))
        global_max = max(np.max(input_2d), np.max(true_2d), np.max(pred_2d))
        
        # 创建子图，保持原始分辨率比例
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
        
        # 输入压力图 - 显示分辨率信息
        im1 = ax1.imshow(input_2d, cmap='viridis', interpolation='nearest',  # 使用nearest避免插值
                        vmin=global_min, vmax=global_max)
        ax1.set_title(f"Input ({input_res}) t={time_step:.2f}", fontsize=12)
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # 真实压力图 - 显示分辨率信息
        im2 = ax2.imshow(true_2d, cmap='viridis', interpolation='nearest',
                        vmin=global_min, vmax=global_max)
        ax2.set_title(f"True ({true_res}) t={time_step:.2f}", fontsize=12)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # 预测压力图 - 显示分辨率信息
        im3 = ax3.imshow(pred_2d, cmap='viridis', interpolation='nearest',
                        vmin=global_min, vmax=global_max)
        ax3.set_title(f"Predicted ({pred_res}) t={time_step:.2f}", fontsize=12)
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
    except Exception as e:
        print(f"绘制图像失败: {e}")
        return

    # 优化布局
    plt.tight_layout(pad=1.0)

    # 保存SVG格式，矢量图形更清晰
    svg_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}.svg")
    try:
        # 使用SVG格式保存参数
        plt.savefig(svg_path, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   format='svg')
        print(f"三联图已保存: {svg_path}")
    except Exception as e:
        print(f"保存SVG失败: {e}")
        # 备选：保存为PNG格式
        try:
            png_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}.png")
            plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
            print(f"备选PNG三联图已保存: {png_path}")
        except Exception as e2:
            print(f"保存备选PNG也失败: {e2}")
    
    plt.close(fig)  # 明确关闭图形对象


def plot_losses(train_losses, val_losses, attention_type, parent_dir="attention_results"):
    """绘制训练和验证损失曲线，优化性能"""
    result_dir = os.path.join(parent_dir, attention_type, "visualization_results")
    os.makedirs(result_dir, exist_ok=True)
    
    # 优化图形创建
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # 使用更高效的绘图方式
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{attention_type} - Training and Validation Loss', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 优化布局
        plt.tight_layout()
        
        # 优先保存PNG格式
        png_path = os.path.join(result_dir, "loss_curves.png")
        plt.savefig(png_path, dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   format='png', optimize=True)
        print(f"损失曲线图已保存: {png_path}")
        
    except Exception as e:
        print(f"绘制损失曲线失败: {e}")
    finally:
        plt.close(fig)


def plot_difference_figure(input_pressure, true_pressure, predicted_pressure, time_step, epoch, attention_type, idx, parent_dir="attention_results", mode="test"):
    """绘制差异图，优化性能"""
    result_dir = os.path.join(parent_dir, attention_type, "visualization_results")
    os.makedirs(result_dir, exist_ok=True)

    # 使用优化的数据处理函数
    def ensure_2d_and_optimize(data, name, max_size=256):
        """快速数据处理，优化性能"""
        if len(data.shape) == 1:
            size = data.shape[0]
            # 快速形状推断
            if size == 16384:
                reshaped = data.reshape(128, 128)
            elif size == 1024:
                reshaped = data.reshape(32, 32)
            else:
                side_len = int(np.sqrt(size))
                if side_len * side_len == size:
                    reshaped = data.reshape(side_len, side_len)
                else:
                    # 简化的因子分解
                    for i in range(int(np.sqrt(size)), 0, -1):
                        if size % i == 0:
                            reshaped = data.reshape(i, size // i)
                            break
                    else:
                        reshaped = data.reshape(1, -1)
        elif len(data.shape) == 2:
            reshaped = data
        else:
            raise ValueError(f"{name} 数据维度不支持: {data.shape}")
        
        # 快速尺寸限制
        h, w = reshaped.shape
        if h > max_size or w > max_size:
            # 使用更高效的下采样
            step_h = max(1, h // max_size)
            step_w = max(1, w // max_size)
            reshaped = reshaped[::step_h, ::step_w]
        
        return reshaped
    
    try:
        # 使用优化的处理函数
        input_2d = ensure_2d_and_optimize(input_pressure, "input_pressure")
        true_2d = ensure_2d_and_optimize(true_pressure, "true_pressure")
        pred_2d = ensure_2d_and_optimize(predicted_pressure, "predicted_pressure")
        
        # 计算差异
        diff = true_2d - pred_2d
    except Exception as e:
        print(f"数据维度处理失败: {e}")
        return

    # 优化图形创建
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 计算统一的colorbar范围
    try:
        # 计算前三个图的全局最小值和最大值，确保colorbar范围一致
        global_min = min(np.min(input_2d), np.min(true_2d), np.min(pred_2d))
        global_max = max(np.max(input_2d), np.max(true_2d), np.max(pred_2d))
        
        # 差异图使用独立的范围，以更好地显示差异
        diff_min = np.min(diff)
        diff_max = np.max(diff)
        
        # 输入压力图
        im1 = axes[0].imshow(input_2d, cmap='viridis', interpolation='bilinear',
                            vmin=global_min, vmax=global_max)  # 统一范围
        axes[0].set_title(f"Input t={time_step:.2f}", fontsize=10)
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # 真实压力图
        im2 = axes[1].imshow(true_2d, cmap='viridis', interpolation='bilinear',
                            vmin=global_min, vmax=global_max)  # 统一范围
        axes[1].set_title(f"True t={time_step:.2f}", fontsize=10)
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # 预测压力图
        im3 = axes[2].imshow(pred_2d, cmap='viridis', interpolation='bilinear',
                            vmin=global_min, vmax=global_max)  # 统一范围
        axes[2].set_title(f"Predicted t={time_step:.2f}", fontsize=10)
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        # 差异图使用独立的范围和不同的colormap
        im4 = axes[3].imshow(diff, cmap='RdBu', interpolation='bilinear',
                            vmin=diff_min, vmax=diff_max)  # 差异图独立范围
        axes[3].set_title(f"Difference t={time_step:.2f}", fontsize=10)
        plt.colorbar(im4, ax=axes[3], shrink=0.8)
        
    except Exception as e:
        print(f"绘制图像失败: {e}")
        return

    # 优化布局
    plt.tight_layout(pad=1.0)

    # 保存SVG格式
    svg_path = os.path.join(result_dir, f"{mode}_diff_epoch_{epoch}_sample_{idx}.svg")
    try:
        plt.savefig(svg_path, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   format='svg')
        print(f"差异图已保存: {svg_path}")
    except Exception as e:
        print(f"保存差异图失败: {e}")
    
    plt.close(fig)


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
    performance_plot = os.path.join(output_dir, f"model_performance_comparison_{timestamp}.svg")
    plot_files.append(performance_plot)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = list(successful_results.keys())
    r2_scores = [successful_results[model].get('final_test_r2', successful_results[model].get('test_r2', 0)) for model in models]
    mse_scores = [successful_results[model].get('final_test_mse', successful_results[model].get('test_mse', 0)) for model in models]
    
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
    plt.savefig(performance_plot.replace('.png', '.svg'), bbox_inches='tight', format='svg')
    plt.close()
    
    # 2. 参数量对比图
    params_plot = os.path.join(output_dir, f"model_parameters_comparison_{timestamp}.svg")
    plot_files.append(params_plot)
    
    param_counts = [successful_results[model].get('num_parameters', successful_results[model].get('parameters', 0)) for model in models]
    
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
    plt.savefig(params_plot.replace('.png', '.svg'), bbox_inches='tight', format='svg')
    plt.close()
    
    # 3. 训练时间对比图
    time_plot = os.path.join(output_dir, f"model_training_time_comparison_{timestamp}.svg")
    plot_files.append(time_plot)
    
    train_times = [successful_results[model].get('train_time', successful_results[model].get('training_time', 0)) for model in models]
    
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
    plt.savefig(time_plot, bbox_inches='tight', format='svg')
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
            f.write(f"| {model_name} | {result.get('final_test_mse', result.get('test_mse', 0)):.6f} | "
                f"{result.get('final_test_r2', result.get('test_r2', 0)):.4f} | {result.get('num_parameters', result.get('parameters', 0)):,} | "
                f"{result.get('training_time', 0):.2f}s |\n")
        
        # 按性能排名
        f.write("\n### 🥇 按性能排名 (MSE越小越好)\n\n")
        sorted_by_mse = sorted(successful_results.items(), key=lambda x: x[1].get('final_test_mse', x[1].get('test_mse', 0)))
        for i, (model_name, result) in enumerate(sorted_by_mse, 1):
            f.write(f"{i}. **{model_name}**: MSE={result.get('final_test_mse', result.get('test_mse', 0)):.6f}\n")
        
        f.write("\n### 🥇 按性能排名 (R²越大越好)\n\n")
        sorted_by_r2 = sorted(successful_results.items(), key=lambda x: x[1].get('final_test_r2', x[1].get('test_r2', 0)), reverse=True)
        for i, (model_name, result) in enumerate(sorted_by_r2, 1):
            f.write(f"{i}. **{model_name}**: R²={result.get('final_test_r2', result.get('test_r2', 0)):.4f}\n")
    
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
    
    loss_plot = os.path.join(output_dir, f"{model_name}_loss_curves_{timestamp}.svg")
    
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
    plt.savefig(loss_plot, bbox_inches='tight', format='svg')
    plt.close()
    
    print(f"✅ {model_name} 损失曲线已保存: {loss_plot}")
    return loss_plot
