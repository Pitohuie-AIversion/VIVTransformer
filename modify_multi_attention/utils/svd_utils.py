import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import os
import glob


def load_difference_matrix(file_path):
    """
    加载差异矩阵（CSV文件）。
    :param file_path: 差异矩阵文件路径
    :return: 加载的差异矩阵
    """
    return np.loadtxt(file_path, delimiter=",")


def perform_svd(difference_matrix):
    """
    对差异矩阵执行SVD分解，返回主成分和奇异值。
    :param difference_matrix: 输入的差异矩阵
    :return: U, S, Vh 作为SVD分解的结果
    """
    U, S, Vh = svd(difference_matrix, full_matrices=False)
    return U, S, Vh


def plot_singular_values(S, file_name):
    """
    绘制奇异值的图像，显示SVD的奇异值分布。
    :param S: 奇异值
    :param file_name: 用于保存图像的文件名
    """
    plt.figure(figsize=(8, 6))
    plt.plot(S, marker='o', linestyle='-', color='b')
    plt.title('Singular Value Decomposition (SVD) - Singular Values')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()


def find_high_energy_regions(S, threshold=0.9):
    """
    根据奇异值的贡献率（例如，设定阈值）找到高能区域。
    :param S: 奇异值
    :param threshold: 用于选择高能区域的贡献阈值（默认 90%）
    :return: 高能区域的索引
    """
    total_energy = np.sum(S)
    energy_ratio = np.cumsum(S) / total_energy  # 计算累计能量比
    high_energy_idx = np.where(energy_ratio >= threshold)[0][0]  # 获取超过阈值的最小索引
    return high_energy_idx, energy_ratio


def save_svd_results(U, S, Vh, output_dir, file_name):
    """
    将SVD分解结果保存到文件。
    :param U: U矩阵
    :param S: 奇异值
    :param Vh: Vh矩阵
    :param output_dir: 输出目录
    :param file_name: 用于保存文件的基名称
    """
    np.savetxt(os.path.join(output_dir, f"{file_name}_U_matrix.csv"), U, delimiter=",")
    np.savetxt(os.path.join(output_dir, f"{file_name}_singular_values.csv"), S, delimiter=",")
    np.savetxt(os.path.join(output_dir, f"{file_name}_Vh_matrix.csv"), Vh, delimiter=",")


def svd_analysis(file_path, threshold=0.9, output_dir="svd_results"):
    """
    完整的SVD分析流程：加载差异矩阵，执行SVD，分析高能区域，保存结果。
    :param file_path: 差异矩阵文件路径
    :param threshold: 高能区域的能量贡献阈值
    :param output_dir: 结果保存目录
    """
    # 加载差异矩阵
    difference_matrix = load_difference_matrix(file_path)

    # 执行SVD
    U, S, Vh = perform_svd(difference_matrix)

    # 获取文件名用于保存结果
    file_name = os.path.basename(file_path).replace(".csv", "")

    # 绘制奇异值并保存图像
    plot_singular_values(S, os.path.join(output_dir, f"{file_name}_singular_values_plot.png"))

    # 查找高能区域
    high_energy_idx, energy_ratio = find_high_energy_regions(S, threshold)

    print(f"High energy region is at index: {high_energy_idx}")
    print(f"Energy ratio for this region: {energy_ratio[high_energy_idx]:.4f}")

    # 保存SVD结果
    save_svd_results(U, S, Vh, output_dir, file_name)

    return high_energy_idx, energy_ratio


def batch_process_svd(input_dir, output_dir, threshold=0.9, summary_file="svd_summary.csv"):
    """
    批量处理指定目录下的所有差异矩阵文件，执行SVD分析，并将结果保存到CSV文件。
    :param input_dir: 差异矩阵文件所在目录
    :param output_dir: 结果保存目录
    :param threshold: 高能区域的能量贡献阈值
    :param summary_file: 汇总结果的文件名
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有差异矩阵文件路径
    files = glob.glob(os.path.join(input_dir, "*.csv"))

    # 打开汇总文件以保存结果
    with open(summary_file, 'w') as f:
        f.write("File, High Energy Region Index, Energy Ratio\n")  # 写入表头

        for file_path in files:
            print(f"Processing file: {file_path}")
            high_energy_idx, energy_ratio = svd_analysis(file_path, threshold, output_dir)
            f.write(f"{os.path.basename(file_path)}, {high_energy_idx}, {energy_ratio[high_energy_idx]:.4f}\n")


if __name__ == "__main__":
    # 设置路径
    input_dir = "F:/Zhaoyang/VIVTransformer/attention_results/relative/difference_results"  # 差异矩阵文件夹路径
    output_dir = "F:/Zhaoyang/VIVTransformer/attention_results/relative/svd_results"  # 结果保存路径
    threshold = 0.9  # 90%的能量阈值
    summary_file = "F:/Zhaoyang/VIVTransformer/attention_results/relative/svd_results/svd_summary.csv"  # 汇总文件路径

    # 批量处理SVD分析
    batch_process_svd(input_dir, output_dir, threshold, summary_file)
