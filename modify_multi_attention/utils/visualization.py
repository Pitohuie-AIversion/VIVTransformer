import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch

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




import os
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_figure(input_pressure, true_pressure, predicted_pressure, reynolds_number, time_step, epoch, idx, attention_type, parent_dir="attention_results", mode="test"):
    # 创建对应注意力机制的子文件夹
    result_dir = os.path.join(parent_dir, attention_type, "visualization_results")
    os.makedirs(result_dir, exist_ok=True)

    # 使用 .detach() 断开计算图
    input_pressure = input_pressure.detach().cpu().numpy() if isinstance(input_pressure, torch.Tensor) else input_pressure
    true_pressure = true_pressure.detach().cpu().numpy() if isinstance(true_pressure, torch.Tensor) else true_pressure
    predicted_pressure = predicted_pressure.detach().cpu().numpy() if isinstance(predicted_pressure, torch.Tensor) else predicted_pressure

    # 保存可视化图像
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

    # 保存图片
    save_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}.png")
    plt.savefig(save_path)
    plt.close()

def plot_difference_figure(true_pressure, predicted_pressure, reynolds_number, time_step, epoch, idx, attention_type, parent_dir="attention_results", mode="test"):
    # 创建对应注意力机制的子文件夹
    result_dir = os.path.join(parent_dir, attention_type, "difference_results")
    os.makedirs(result_dir, exist_ok=True)

    # 计算差异（绝对误差）
    difference = np.abs(true_pressure - predicted_pressure)

    # 保存文件名中带上Reynolds Number 和 Time Step
    save_prefix = f"Re_{reynolds_number}_time_{time_step:.2f}_{mode}_epoch_{epoch}_sample_{idx}"

    # 根据 config.save_format 选择 npy/csv
    save_fmt = "npy"  # 默认为npy格式
    npy_save_path = os.path.join(result_dir, f"{save_prefix}_difference_matrix.npy")
    np.save(npy_save_path, difference)

    # 保存为CSV格式
    csv_save_path = os.path.join(result_dir, f"{save_prefix}_difference_matrix.csv")
    np.savetxt(csv_save_path, difference, delimiter=",")  # 保存为 CSV 格式

    # 绘制差异图
    plt.figure(figsize=(6, 5))
    plt.imshow(difference, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Difference (|True - Predicted|) at t={time_step:.2f}")

    plt.tight_layout()

    # 保存图片到对应文件夹
    save_path = os.path.join(result_dir, f"{save_prefix}_difference.png")
    plt.savefig(save_path)
    plt.close()
