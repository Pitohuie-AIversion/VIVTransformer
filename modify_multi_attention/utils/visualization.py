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
