import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import svd
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity


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


def weight_first_mode(U, weight_factor=1.0):
    """
    加权SVD模态1（U的第一列），并返回加权后的特征向量。
    :param U: 左奇异向量矩阵
    :param weight_factor: 加权因子
    :return: 加权后的特征向量
    """
    first_mode = U[:, 0]  # 提取模态1（U矩阵的第一列）
    weighted_first_mode = first_mode * weight_factor  # 对模态1加权
    return weighted_first_mode


def map_to_coordinates(weighted_first_mode, difference_matrix):
    """
    将加权后的特征向量映射到差异矩阵的空间位置。
    :param weighted_first_mode: 加权后的特征向量
    :param difference_matrix: 差异矩阵
    :return: 映射后的坐标值
    """
    # 假设加权后的特征向量将映射到差异矩阵的位置上
    coords = np.dot(difference_matrix, weighted_first_mode)  # 将加权模态与差异矩阵相乘，得到坐标值
    return coords


def plot_weighted_mode(weighted_first_mode, file_name):
    """
    绘制加权后的模态1。
    :param weighted_first_mode: 加权后的特征向量
    :param file_name: 用于保存图像的文件名
    """
    plt.figure(figsize=(8, 6))
    plt.plot(weighted_first_mode)
    plt.title('Weighted Mode 1 (First Singular Vector)')
    plt.xlabel('Index')
    plt.ylabel('Weighted Value')
    plt.savefig(file_name)
    plt.show()


def calculate_cosine_similarity_for_first_modes(file_list):
    """
    计算每个文件第一个特征向量的余弦相似度矩阵。
    :param file_list: 文件列表
    :return: 相似度矩阵
    """
    first_modes = []
    for file_path in file_list:
        # 读取每个文件的差异矩阵
        difference_matrix = load_difference_matrix(file_path)
        # 执行SVD
        U, _, _ = perform_svd(difference_matrix)
        # 提取第一个特征向量（U的第一列）
        first_modes.append(U[:, 0])

    # 计算第一个特征向量之间的余弦相似度
    similarity_matrix = cosine_similarity(first_modes)
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, file_list, file_name):
    """
    绘制余弦相似度矩阵的热力图。
    :param similarity_matrix: 相似度矩阵
    :param file_list: 文件列表
    :param file_name: 用于保存图像的文件名
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Cosine Similarity Matrix of First Singular Vectors')

    # 添加文件标签
    plt.xticks(np.arange(len(file_list)), [os.path.basename(f) for f in file_list], rotation=90)
    plt.yticks(np.arange(len(file_list)), [os.path.basename(f) for f in file_list])

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


def find_high_energy_region(difference_matrix):
    """
    根据差异矩阵找到高能区域（最大差异值的位置）。
    :param difference_matrix: 差异矩阵
    :return: 高能区域的坐标 (x, y)
    """
    # 获取差异矩阵的最大值及其索引
    max_idx = np.argmax(difference_matrix)  # 获取最大值的位置索引
    y, x = np.unravel_index(max_idx, difference_matrix.shape)  # 转换为二维坐标
    return x, y


def plot_difference_figure(difference_matrix, time_step, epoch, attention_type, idx, parent_dir="attention_results",
                           mode="test", high_energy_region=None, radius=5):
    """
    绘制差异图并用圆形框选高能区域。展示流场的差异。
    :param difference_matrix: 差异矩阵
    :param time_step: 当前时间步
    :param epoch: 当前训练轮数
    :param attention_type: 使用的注意力类型
    :param idx: 当前样本索引
    :param parent_dir: 保存结果的父目录
    :param mode: 模式（train 或 test）
    :param high_energy_region: 高能区域的坐标，格式为 (x, y)
    :param radius: 圆的半径，用来显示高能区域
    """
    # 创建对应注意力机制的子文件夹
    result_dir = os.path.join(parent_dir, attention_type, "difference_results")
    os.makedirs(result_dir, exist_ok=True)

    # 打印差异矩阵的统计信息，调试
    print(
        f"Difference matrix stats: min={np.min(difference_matrix)}, max={np.max(difference_matrix)}, mean={np.mean(difference_matrix)}")

    # 绘制差异图，表示流场的差异
    plt.figure(figsize=(6, 5))
    img = plt.imshow(difference_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(img)
    plt.title(f"Difference (|True - Predicted|) at t={time_step:.2f}")

    # 绘制高能区域（如果有的话）
    if high_energy_region is not None:
        x, y = high_energy_region
        circle = patches.Circle((x, y), radius=radius, linewidth=2, edgecolor='r', facecolor='none', linestyle='--')
        plt.gca().add_patch(circle)  # 将圆形框添加到图像上

    plt.tight_layout()

    # 保存图片到对应文件夹
    save_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}_difference.png")
    plt.savefig(save_path)
    plt.show()  # 显示图形


def svd_analysis(file_path, threshold=0.9, output_dir="svd_results", top_n=10, weight_factor=1.0):
    """
    完整的SVD分析流程：加载差异矩阵，执行SVD，分析高能区域，保存结果。
    :param file_path: 差异矩阵文件路径
    :param threshold: 高能区域的能量贡献阈值
    :param output_dir: 结果保存目录
    :param top_n: 显示前N个特征向量
    :param weight_factor: 加权因子
    """
    # 加载差异矩阵
    difference_matrix = load_difference_matrix(file_path)

    # 执行SVD
    U, S, Vh = perform_svd(difference_matrix)

    # 对第一个特征向量进行加权
    weighted_first_mode = weight_first_mode(U, weight_factor)

    # 绘制加权后的模态1
    plot_weighted_mode(weighted_first_mode, os.path.join(output_dir,
                                                         f"{os.path.basename(file_path).replace('.csv', '')}_weighted_first_mode.png"))

    # 获取文件名用于保存结果
    file_name = os.path.basename(file_path).replace(".csv", "")

    # 找到高能区域
    high_energy_region = find_high_energy_region(difference_matrix)

    # 绘制差异图并框选高能区域
    plot_difference_figure(difference_matrix, time_step=0.1, epoch=10, attention_type="relative", idx=0,
                           parent_dir="attention_results", mode="test", high_energy_region=high_energy_region)

    # 计算余弦相似度矩阵并可视化
    similarity_matrix = calculate_cosine_similarity_for_first_modes([file_path])
    plot_similarity_matrix(similarity_matrix, [file_path],
                           os.path.join(output_dir, f"{file_name}_similarity_matrix.png"))

    return high_energy_region


def batch_process_svd(input_dir, output_dir, threshold=0.9, summary_file="svd_summary.csv", top_n=10,
                      weight_factor=1.0):
    """
    批量处理指定目录下的所有差异矩阵文件，执行SVD分析，并将结果保存到CSV文件。
    :param input_dir: 差异矩阵文件所在目录
    :param output_dir: 结果保存目录
    :param threshold: 高能区域的能量贡献阈值
    :param summary_file: 汇总结果的文件名
    :param top_n: 显示前N个特征向量
    :param weight_factor: 加权因子
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有差异矩阵文件路径
    files = glob.glob(os.path.join(input_dir, "*.csv"))

    # 计算所有CSV文件的第一个特征向量之间的相似度矩阵
    similarity_matrix = calculate_cosine_similarity_for_first_modes(files)

    # 绘制相似度矩阵
    plot_similarity_matrix(similarity_matrix, files, os.path.join(output_dir, "similarity_matrix.png"))

    # 打开汇总文件以保存结果
    with open(summary_file, 'w') as f:
        f.write("File, High Energy Region X, High Energy Region Y\n")  # 写入表头

        for file_path in files:
            print(f"Processing file: {file_path}")
            high_energy_region = svd_analysis(file_path, threshold, output_dir, top_n, weight_factor)
            f.write(f"{os.path.basename(file_path)}, {high_energy_region[0]}, {high_energy_region[1]}\n")


if __name__ == "__main__":
    # 设置路径
    input_dir = "F:/Zhaoyang/VIVTransformer/attention_results/relative/difference_results"  # 差异矩阵文件夹路径
    output_dir = "F:/Zhaoyang/VIVTransformer/attention_results/relative/svd_results"  # 结果保存路径
    threshold = 0.9  # 90%的能量阈值
    summary_file = "F:/Zhaoyang/VIVTransformer/attention_results/relative/svd_results/svd_summary.csv"  # 汇总文件路径
    top_n = 10  # 显示前10个特征向量
    weight_factor = 1.5  # 模态加权因子

    # 批量处理SVD分析
    batch_process_svd(input_dir, output_dir, threshold, summary_file, top_n, weight_factor)
