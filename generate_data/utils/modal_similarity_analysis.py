#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模态相似度分析脚本：
- 加载 SVDModalProjector 的保存文件 (svd_projector.pkl)
- 计算并可视化：
  1) 输入模态之间的余弦相似度 (V_in vs V_in)
  2) 输出模态之间的余弦相似度 (V_out vs V_out)
  3) 输入与输出模态之间的余弦相似度 (V_in vs V_out)
  4) 输入与输出时间系数的相关矩阵 (U_in vs U_out)
- 保存热力图 (PNG/SVG) 与统计摘要 (JSON/MD)
"""

import os
import json
import argparse
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_n = _row_normalize(A)
    B_n = _row_normalize(B)
    return A_n @ B_n.T


def correlation_matrix_columns(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """逐列计算相关系数：corr(i,j) = corrcoef(X[:,i], Y[:,j])"""
    n_x = X.shape[1]
    n_y = Y.shape[1]
    C = np.zeros((n_x, n_y), dtype=np.float64)
    for i in range(n_x):
        x = X[:, i]
        x = (x - x.mean()) / (x.std() + 1e-12)
        for j in range(n_y):
            y = Y[:, j]
            y = (y - y.mean()) / (y.std() + 1e-12)
            C[i, j] = float(np.mean(x * y))
    return C


def plot_heatmap(mat: np.ndarray, title: str, save_png: str, save_svg: str, vmin: float = -1.0, vmax: float = 1.0):
    os.makedirs(os.path.dirname(save_png), exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.0)
    import matplotlib
    # 统一使用全局 sitecustomize.py 的中文字体和负号设置
# matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
# matplotlib.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(mat, vmin=vmin, vmax=vmax, cmap="coolwarm", annot=False,
                     xticklabels=[f"{i+1}" for i in range(mat.shape[1])],
                     yticklabels=[f"{i+1}" for i in range(mat.shape[0])])
    ax.set_title(title)
    ax.set_xlabel("模式")
    ax.set_ylabel("模式")
    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.savefig(save_svg)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SVD 模态相似度分析")
    parser.add_argument("--projector", type=str, required=True, help="svd_projector.pkl 路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认与projector同目录）")
    parser.add_argument("--top-k", type=int, default=10, help="参与相似度分析的前K个模态")
    args = parser.parse_args()

    projector_path = args.projector
    out_dir = args.output_dir or os.path.dirname(projector_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(projector_path, "rb") as f:
        save_data = pickle.load(f)

    V_in = save_data.get("input_V")  
    V_out = save_data.get("output_V") 
    U_in = save_data.get("input_projector") 
    U_out = save_data.get("output_projector") 
    sing_in = save_data.get("input_singular_values")
    sing_out = save_data.get("output_singular_values")

    if V_in is None or V_out is None:
        raise RuntimeError("svd_projector.pkl 中缺少 V 矩阵，无法进行相似度分析")

    V_in = np.asarray(V_in)
    V_out = np.asarray(V_out)
    print(f"V_in 形状: {V_in.shape}, V_out 形状: {V_out.shape}")
    
    k = int(min(args.top_k, V_in.shape[0], V_out.shape[0]))
    V_in_k = V_in[:k, :]
    V_out_k = V_out[:k, :]
    
    if U_in is not None:
        U_in = np.asarray(U_in)
        U_in_k = U_in[:, :k]
    else:
        U_in_k = None
        
    if U_out is not None:
        U_out = np.asarray(U_out)
        U_out_k = U_out[:, :k]
    else:
        U_out_k = None

    # 1) 余弦相似度矩阵
    S_in_in = cosine_similarity_matrix(V_in_k, V_in_k)
    S_out_out = cosine_similarity_matrix(V_out_k, V_out_k)
    
    # 注意：由于输入和输出V矩阵的特征维度不同，无法直接计算跨域相似度
    # 但我们可以通过时间系数U来计算跨域相关性
    print(f"警告：输入V({V_in_k.shape[1]})和输出V({V_out_k.shape[1]})特征维度不同，跳过V_in vs V_out相似度计算")
    S_in_out = None

    # 2) 时间系数相关矩阵
    C_u = None
    if U_in_k is not None and U_out_k is not None:
        C_u = correlation_matrix_columns(U_in_k, U_out_k)

    # 保存热力图
    plot_heatmap(S_in_in, f"输入模态余弦相似度 (Top-{k})", os.path.join(out_dir, "sim_input_input.png"), os.path.join(out_dir, "sim_input_input.svg"))
    plot_heatmap(S_out_out, f"输出模态余弦相似度 (Top-{k})", os.path.join(out_dir, "sim_output_output.png"), os.path.join(out_dir, "sim_output_output.svg"))
    if S_in_out is not None:
        plot_heatmap(S_in_out, f"输入-输出 模态余弦相似度 (Top-{k})", os.path.join(out_dir, "sim_input_output.png"), os.path.join(out_dir, "sim_input_output.svg"))
    if C_u is not None:
        plot_heatmap(C_u, f"输入-输出 时间系数相关 (Top-{k})", os.path.join(out_dir, "corr_U_input_output.png"), os.path.join(out_dir, "corr_U_input_output.svg"))

    # 统计摘要
    def top_matches(sim_mat: np.ndarray, topn: int = 1):
        res = []
        for i in range(sim_mat.shape[0]):
            j = int(np.argmax(np.abs(sim_mat[i, :])))
            val = float(sim_mat[i, j])
            res.append({"i": int(i+1), "best_j": int(j+1), "similarity": val, "abs_similarity": float(abs(val))})
        res.sort(key=lambda x: -x["abs_similarity"])  # 按绝对相似度排序
        return res[:min(topn, len(res))]

    offdiag_rms_in = float(np.sqrt(np.mean((S_in_in - np.eye(k))**2)))
    offdiag_rms_out = float(np.sqrt(np.mean((S_out_out - np.eye(k))**2)))

    summary = {
        "top_k": k,
        "input_singular_values": sing_in[:k].tolist() if isinstance(sing_in, (list, np.ndarray)) else None,
        "output_singular_values": sing_out[:k].tolist() if isinstance(sing_out, (list, np.ndarray)) else None,
        "offdiag_rms_input_input": offdiag_rms_in,
        "offdiag_rms_output_output": offdiag_rms_out,
        "best_matches_input_to_output": top_matches(S_in_out, topn=k) if S_in_out is not None else [],
        "has_cross_domain_V_similarity": S_in_out is not None,
        "has_cross_domain_U_correlation": C_u is not None,
        "notes": "余弦相似度范围[-1,1]，越接近±1表示越强的方向一致性；输入/输出内部相似度应接近单位阵。若输入输出特征维度不同，将跳过V的跨域相似度计算，改用U的相关性。"
    }

    with open(os.path.join(out_dir, "modal_similarity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 也输出一份简短Markdown
    md_lines = [
        f"# 模态相似度分析摘要 (Top-{k})",
        "",
        f"- 输入内部相似度(与单位阵差异) RMS: {offdiag_rms_in:.3e}",
        f"- 输出内部相似度(与单位阵差异) RMS: {offdiag_rms_out:.3e}",
        f"- 最佳匹配(输入->输出)前{min(5,k)}条：",
    ]
    for item in summary["best_matches_input_to_output"][:min(5,k)]:
        md_lines.append(f"  - 输入模态 {item['i']} ↔ 输出模态 {item['best_j']} | 余弦相似度 = {item['similarity']:.4f}")
    with open(os.path.join(out_dir, "modal_similarity_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("相似度分析完成，结果保存在:", out_dir)


if __name__ == "__main__":
    main()