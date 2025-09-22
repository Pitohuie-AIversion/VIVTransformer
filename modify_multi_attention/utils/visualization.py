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

    # 确保所有输入都是2D数组，并限制图像尺寸
    def ensure_2d_and_limit_size(data, name, max_size=512):
        """确保数据是2D格式，并限制最大尺寸以避免matplotlib错误"""
        if len(data.shape) == 1:
            # 1D数据，尝试重塑为2D
            size = data.shape[0]
            side_len = int(np.sqrt(size))
            if side_len * side_len == size:
                reshaped = data.reshape(side_len, side_len)
            else:
                # 无法重塑为正方形，尝试找到合适的矩形形状
                if size == 16384:
                    reshaped = data.reshape(128, 128)
                elif size == 1024:
                    reshaped = data.reshape(32, 32)
                else:
                    # 尝试找到合适的因子分解
                    factors = []
                    for i in range(1, int(np.sqrt(size)) + 1):
                        if size % i == 0:
                            factors.append((i, size // i))
                    
                    if factors:
                        # 选择最接近正方形的形状
                        h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                        reshaped = data.reshape(h, w)
                    else:
                        # 最后的备选方案：重塑为行向量
                        reshaped = data.reshape(1, -1)
        elif len(data.shape) == 2:
            reshaped = data
        else:
            raise ValueError(f"{name} 数据维度不支持: {data.shape}")
        
        # 检查并限制图像尺寸
        h, w = reshaped.shape
        if h > max_size or w > max_size:
            print(f"Warning: {name} 图像尺寸 {h}x{w} 过大，将进行下采样到 {max_size}x{max_size}")
            
            # 计算下采样比例
            scale_h = max_size / h if h > max_size else 1
            scale_w = max_size / w if w > max_size else 1
            scale = min(scale_h, scale_w)
            
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # 使用简单的下采样方法
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            
            reshaped = reshaped[::step_h, ::step_w]
            print(f"  下采样后尺寸: {reshaped.shape}")
        
        # 最终安全检查：确保尺寸不超过限制
        h, w = reshaped.shape
        if h > 65535 or w > 65535:
            print(f"Error: {name} 图像尺寸 {h}x{w} 仍然过大，强制裁剪到 512x512")
            reshaped = reshaped[:512, :512]
        
        return reshaped
    
    try:
        input_2d = ensure_2d_and_limit_size(input_pressure, "input_pressure")
        true_2d = ensure_2d_and_limit_size(true_pressure, "true_pressure")
        pred_2d = ensure_2d_and_limit_size(predicted_pressure, "predicted_pressure")
    except Exception as e:
        print(f"数据维度处理失败: {e}")
        return

    # 创建图形，使用合理的尺寸
    plt.figure(figsize=(18, 5))

    # 输入压力图
    plt.subplot(1, 3, 1)
    try:
        plt.imshow(input_2d, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Input Pressure Matrix at t={time_step:.2f}")
    except Exception as e:
        print(f"绘制输入图像失败: {e}")
        plt.text(0.5, 0.5, f"输入图像绘制失败\n{str(e)}", ha='center', va='center', transform=plt.gca().transAxes)

    # 真实压力图
    plt.subplot(1, 3, 2)
    try:
        plt.imshow(true_2d, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title(f"True Pressure Matrix at t={time_step:.2f}")
    except Exception as e:
        print(f"绘制真实图像失败: {e}")
        plt.text(0.5, 0.5, f"真实图像绘制失败\n{str(e)}", ha='center', va='center', transform=plt.gca().transAxes)

    # 预测压力图
    plt.subplot(1, 3, 3)
    try:
        plt.imshow(pred_2d, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Predicted Pressure Matrix at t={time_step:.2f}")
    except Exception as e:
        print(f"绘制预测图像失败: {e}")
        plt.text(0.5, 0.5, f"预测图像绘制失败\n{str(e)}", ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()

    # 仅保存SVG矢量格式，减少磁盘占用
    svg_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}.svg")
    try:
        plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
        print(f"三联图已保存: {svg_path}")
    except Exception as e:
        print(f"保存三联图失败: {e}")
        # 尝试保存为PNG格式作为备选
        png_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}.png")
        try:
            plt.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=150)
            print(f"三联图已保存为PNG: {png_path}")
        except Exception as e2:
            print(f"保存PNG格式也失败: {e2}")
    
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
    plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 参数量对比图
    params_plot = os.path.join(output_dir, f"model_parameters_comparison_{timestamp}.png")
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
    plt.savefig(params_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 训练时间对比图
    time_plot = os.path.join(output_dir, f"model_training_time_comparison_{timestamp}.png")
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
