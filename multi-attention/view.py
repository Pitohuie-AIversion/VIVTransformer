from dataset import PressureDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 创建结果文件夹
result_folder = "visualization_results"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 加载数据集
dataset = PressureDataset(merged_file_path="F:/Zhaoyang/merged_all_pressures_separated_normalized.pt")
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 获取一个样本
for in_press_flat, out_press_flat, _ in data_loader:
    input_matrix = in_press_flat.view(20, 20).numpy().squeeze()
    output_matrix = out_press_flat.view(200, 200).numpy().squeeze()
    break

# 可视化函数并保存图片
def visualize_preprocessing(input_matrix, output_matrix, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(input_matrix, cmap='coolwarm')
    axes[0, 0].set_title('Original Input Matrix (20x20)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')

    axes[0, 1].plot(input_matrix.flatten())
    axes[0, 1].set_title('Flattened Input Vector (400)')
    axes[0, 1].set_xlabel('Index')

    axes[1, 0].imshow(output_matrix, cmap='coolwarm')
    axes[1, 0].set_title('Original Output Matrix (200x200)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')

    axes[1, 1].plot(output_matrix.flatten())
    axes[1, 1].set_title('Flattened Output Vector (40000)')
    axes[1, 1].set_xlabel('Index')

    plt.tight_layout()
    plt.savefig(save_path,dpi=1000)
    plt.close()

# 保存图片
image_save_path = os.path.join(result_folder, 'preprocessing_visualization.png')
visualize_preprocessing(input_matrix, output_matrix, image_save_path)

print(f"Visualization saved at {image_save_path}")
