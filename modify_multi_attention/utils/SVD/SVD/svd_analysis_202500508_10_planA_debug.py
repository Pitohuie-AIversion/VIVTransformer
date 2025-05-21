import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import svd
import os
import glob
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
    os.makedirs(output_dir, exist_ok=True)
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
                      summary_file="svd_summary.csv", top_n=10,
                      weight_factor=1.0, num_modes=10):
    # 0. 收集所有差异图文件
    files = glob.glob(os.path.join(input_dir, "*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    # 1. 从文件名解析出 re, time, iter 列表
    pattern = re.compile(
        r"Re_(?P<re>\d+)_time_(?P<time>[\d\.]+)_train_epoch_(?P<it>\d+)_sample_(?P<sample>\d+)_difference_matrix\.npy$"
    )
    matched = []
    unmatched = []
    for f in files:
        m = pattern.match(os.path.basename(f))
        if m:
            matched.append((f, m))
        else:
            unmatched.append(f)
    if not matched:
        raise RuntimeError("No files match the expected pattern.")
    # 打印未匹配文件
    if unmatched:
        print("⚠️ Unmatched files (will be skipped):")
        for f in unmatched:
            print("  ", f)

    re_list   = sorted({int(m.group('re'))   for _, m in matched})
    t_list    = sorted({float(m.group('time')) for _, m in matched})
    iter_list = sorted({int(m.group('it'))   for _, m in matched})

    # 2. 初始化多模态权重图容器：mode_weights[mode][i_re][i_time][i_iter]
    mode_weights = [
        [[[None for _ in iter_list] for _ in t_list]
           for _ in re_list]
        for _ in range(num_modes)
    ]

    # 3. 全局相似度并导出 CSV
    os.makedirs(output_dir, exist_ok=True)
    sim_all = calculate_cosine_similarity_for_first_modes([f for f, _ in matched])
    np.savetxt(os.path.join(output_dir, "similarity_all.csv"), sim_all, delimiter=",")
    plot_similarity_matrix(sim_all, [f for f, _ in matched],
                           os.path.join(output_dir, "similarity_matrix_all.png"))

    # 4. 按 Reynolds 分组分析
    re_dir = os.path.join(output_dir, "by_Re")
    os.makedirs(re_dir, exist_ok=True)
    for r in re_list:
        group = [f for f, m in matched if int(m.group('re')) == r]
        if len(group) > 1:
            sim = calculate_cosine_similarity_for_first_modes(group)
            np.savetxt(os.path.join(re_dir, f"Re_{r}_sim.csv"), sim, delimiter=",")
            plot_similarity_matrix(sim, group, os.path.join(re_dir, f"Re_{r}_sim.png"))

    # 5. 按 时间 分组分析
    time_dir = os.path.join(output_dir, "by_time")
    os.makedirs(time_dir, exist_ok=True)
    for t in t_list:
        group = [f for f, m in matched if float(m.group('time')) == t]
        if len(group) > 1:
            sim = calculate_cosine_similarity_for_first_modes(group)
            np.savetxt(os.path.join(time_dir, f"time_{t}_sim.csv"), sim, delimiter=",")
            plot_similarity_matrix(sim, group, os.path.join(time_dir, f"time_{t}_sim.png"))

    # 6. 单文件高能区 & 多模态权重采集
    with open(os.path.join(output_dir, summary_file), 'w') as fsum:
        fsum.write("File,HighEnergyX,HighEnergyY\n")
        for fp, m in matched:
            re_idx  = int(m.group('re'))
            t_idx   = float(m.group('time'))
            it_idx  = int(m.group('it'))
            i = re_list.index(re_idx)
            j = t_list.index(t_idx)
            k = iter_list.index(it_idx)

            # 高能区分析
            x, y = svd_analysis(fp, threshold, output_dir, top_n, weight_factor)
            fsum.write(f"{os.path.basename(fp)},{x},{y}\n")

            # 提取每个模式的权重图
            diff = load_difference_matrix(fp)
            U, S, Vh = perform_svd(diff)
            for mode in range(num_modes):
                u = U[:, mode]
                v = Vh[mode, :]
                w2d = np.abs(np.outer(u, v))
                w2d = w2d / (w2d.max() + 1e-12)
                # 转成 Tensor
                mode_weights[mode][i][j][k] = torch.from_numpy(w2d.astype(np.float32))

    # 7. 检查并填充 None，再保存每个模式的权重
    # 获取 H、W 大小
    sample_diff = load_difference_matrix(matched[0][0])
    H, W = sample_diff.shape
    zero_tensor = torch.zeros((H, W), dtype=torch.float32)
    for mode in range(num_modes):
        for i in range(len(re_list)):
            for j in range(len(t_list)):
                for k in range(len(iter_list)):
                    if mode_weights[mode][i][j][k] is None:
                        print(f"⚠️ Filling missing: mode{mode+1}, Re={re_list[i]}, time={t_list[j]}, iter={iter_list[k]}")
                        mode_weights[mode][i][j][k] = zero_tensor.clone()

        mode_dir = os.path.join(output_dir, f"mode{mode+1}")
        os.makedirs(mode_dir, exist_ok=True)
        out_path = os.path.join(mode_dir, f"svd_mode{mode+1}_weights.pt")
        torch.save(mode_weights[mode], out_path)
        print(f"Saved mode {mode+1} weights to {out_path}")

if __name__ == "__main__":
    batch_process_svd(
        input_dir=r"F:\Zhaoyang\VIVTransformer_svd_analyse\attention_results\bam\difference_results",
        output_dir=r"F:\Zhaoyang\VIVTransformer_svd_analyse\attention_results_mask_used\bam\svd_results",
        summary_file="svd_summary.csv",
        top_n=10,
        weight_factor=1.5,
        num_modes=10
    )
