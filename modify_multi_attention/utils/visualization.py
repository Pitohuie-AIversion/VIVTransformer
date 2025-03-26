import os
import matplotlib.pyplot as plt
import numpy as np
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

    # 保存图片到对应文件夹
    save_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}.png")
    plt.savefig(save_path)
    plt.close()


def plot_losses(train_loss, valid_loss, test_loss, save_path=None):
    import matplotlib.pyplot as plt
    if len(train_loss) == 0 or len(valid_loss) == 0 or len(test_loss) == 0:
        print("⚠️ 损失列表为空，无法绘制Loss曲线！")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Loss曲线已保存到: {save_path}")
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

    # 保存图片到对应文件夹
    save_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}_difference.png")
    plt.savefig(save_path)
    plt.close()
