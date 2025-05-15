import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import svd
import os
glob = __import__('glob')
import re
from sklearn.metrics.pairwise import cosine_similarity
import torch  # 用于保存 .pt

def load_difference_matrix(file_path):
    return np.load(file_path)

def perform_svd(difference_matrix):
    return svd(difference_matrix, full_matrices=False)

def plot_weighted_mode(weighted_first_mode, file_name):
    plt.figure(figsize=(8, 6))
    plt.plot(weighted_first_mode)
    plt.title('Weighted Mode 1 (First Singular Vector)')
    plt.xlabel('Index')
    plt.ylabel('Weighted Value')
    plt.savefig(file_name, dpi=300)
    plt.close()

def plot_difference_figure(difference_matrix, time_step, epoch, attention_type, idx,
                           parent_dir="attention_results", mode="test",
                           high_energy_region=None, radius=5):
    result_dir = os.path.join(parent_dir, attention_type, "difference_results")
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(difference_matrix, cmap='hot', interpolation='nearest')
    plt.axis('off')
    if high_energy_region is not None:
        x, y = high_energy_region
        circle = patches.Circle((x, y), radius=radius, linewidth=2,
                                edgecolor='r', facecolor='none', linestyle='--')
        plt.gca().add_patch(circle)
    save_path = os.path.join(result_dir,
                             f"{mode}_epoch_{epoch}_sample_{idx}_difference.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def calculate_cosine_similarity_for_first_modes(file_list):
    first_modes = []
    for file_path in file_list:
        diff = load_difference_matrix(file_path)
        U, _, _ = perform_svd(diff)
        first_modes.append(U[:, 0])
    return cosine_similarity(first_modes)

def plot_similarity_matrix(similarity_matrix, file_list, file_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_yticks([])
    cbar.outline.set_visible(False)
    plt.tight_layout(pad=0)
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def svd_analysis(file_path, threshold=0.9, output_dir="svd_results", top_n=10, weight_factor=1.0):
    diff = load_difference_matrix(file_path)
    U, S, Vh = perform_svd(diff)
    # 高能区原有分析
    weighted = np.abs(U[:, 0]) * weight_factor
    plot_weighted_mode(weighted, os.path.join(output_dir,
                                              os.path.basename(file_path).replace('.npy', '_mode1.png')))
    idx = np.argmax(diff)
    y, x = np.unravel_index(idx, diff.shape)
    plot_difference_figure(diff, time_step=0.1, epoch=10,
                           attention_type="bam", idx=0,
                           parent_dir=os.path.dirname(os.path.dirname(file_path)),
                           high_energy_region=(x, y))
    return x, y

def batch_process_svd(input_dir, output_dir, threshold=0.9,
                      summary_file="svd_summary.csv", top_n=10, weight_factor=1.0,
                      num_modes=10):
    files = glob.glob(os.path.join(input_dir, "*.npy"))
    os.makedirs(output_dir, exist_ok=True)

    # —— 新增：解析文件名以获取所有 re 和 time 列表 ——
    pattern = re.compile(r"Re_(?P<re>\d+)_time_(?P<time>\d+)\.npy$")
    matched = [m for m in (pattern.match(os.path.basename(f)) for f in files) if m]
    re_list = sorted({int(m.group('re')) for m in matched})
    t_list  = sorted({int(m.group('time')) for m in matched})

    # 初始化多模态权重图容器：mode_weights[mode][i][j] = Tensor(H,W)
    mode_weights = [
        [[None for _ in t_list] for _ in re_list]
        for _ in range(num_modes)
    ]

    # 1. 全局相似度并导出 CSV
    sim_all = calculate_cosine_similarity_for_first_modes(files)
    np.savetxt(os.path.join(output_dir, "similarity_all.csv"), sim_all, delimiter=",")
    plot_similarity_matrix(sim_all, files, os.path.join(output_dir, "similarity_matrix_all.png"))

    # 2. 按 Reynolds 和 时间 分组横向分析
    # ——— 同原代码 ———
    matched = [m for m in (pattern.match(os.path.basename(f)) for f in files) if m]
    unique_res   = sorted({m.group('re') for m in matched})
    unique_times = sorted({m.group('time') for m in matched})

    # 按 Re 分组
    re_dir = os.path.join(output_dir, "by_Re")
    os.makedirs(re_dir, exist_ok=True)
    for r in unique_res:
        group = [f for f in files if f"Re_{r}_" in os.path.basename(f)]
        if len(group) > 1:
            sim = calculate_cosine_similarity_for_first_modes(group)
            np.savetxt(os.path.join(re_dir, f"Re_{r}_sim.csv"), sim, delimiter=",")
            plot_similarity_matrix(sim, group, os.path.join(re_dir, f"Re_{r}_sim.png"))

    # 按 时间 分组
    time_dir = os.path.join(output_dir, "by_time")
    os.makedirs(time_dir, exist_ok=True)
    for t in unique_times:
        group = [f for f in files if f"time_{t}" in os.path.basename(f)]
        if len(group) > 1:
            sim = calculate_cosine_similarity_for_first_modes(group)
            np.savetxt(os.path.join(time_dir, f"time_{t}_sim.csv"), sim, delimiter=",")
            plot_similarity_matrix(sim, group, os.path.join(time_dir, f"time_{t}_sim.png"))

    # 3. 单文件高能区汇总 & 多模态权重采集
    with open(summary_file, 'w') as f:
        f.write("File,HighEnergyX,HighEnergyY\n")
        for fp in files:
            m = pattern.search(os.path.basename(fp))
            if not m:
                continue
            re_idx = int(m.group('re'))
            t_idx  = int(m.group('time'))
            i = re_list.index(re_idx)
            j = t_list.index(t_idx)

            # 原有高能区分析
            x, y = svd_analysis(fp, threshold, output_dir, top_n, weight_factor)
            f.write(f"{os.path.basenamec(fp)},{x},{y}\n")

            # —— 新增：提取前 num_modes 个模态的权重图 ——
            diff = load_difference_matrix(fp)
            U, S, Vh = perform_svd(diff)
            for mode in range(num_modes):
                u = U[:, mode]
                v = Vh[mode, :]
                w2d = np.abs(np.outer(u, v))
                w2d = w2d / (w2d.max() + 1e-12)
                mode_weights[mode][i][j] = torch.from_numpy(w2d.astype(np.float32))

    # 保存每个模式的权重图到 .pt 文件
    for mode in range(num_modes):
        out_path = os.path.join(output_dir, f"svd_mode{mode+1}_weights.pt")
        torch.save(mode_weights[mode], out_path)
        print(f"Saved mode {mode+1} weights to {out_path}")

if __name__ == "__main__":
    input_dir = r"F:\Zhaoyang\VIVTransformer_svd_analyse\attention_results\bam\difference_results"
    output_dir = r"F:\Zhaoyang\VIVTransformer_svd_analyse\attention_results\bam\svd_results"
    summary_file = os.path.join(output_dir, "svd_summary.csv")
    batch_process_svd(
        input_dir, output_dir,
        threshold=0.9,
        summary_file=summary_file,
        top_n=10,
        weight_factor=1.5,
        num_modes=10  # 提取前10个主成分模式
    )
