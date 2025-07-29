import numpy as np
import yaml

def generate_loss_configs_topk10(
    base_weight_range=(0.7, 0.9),
    num_configs=None,
    base_weight_step=0.01,
    topk=10,
    distribution="dirichlet",  # 可选：dirichlet / gaussian / uniform
    seed=42,
    output_path="config_topk10_auto.yaml"
):
    np.random.seed(seed)
    configs = []

    # 自动生成 base_weight 序列
    if num_configs is not None:
        base_weights = np.linspace(base_weight_range[0], base_weight_range[1], num_configs)
    else:
        base_weights = np.arange(base_weight_range[0], base_weight_range[1] + 1e-8, base_weight_step)

    for idx, bw in enumerate(base_weights):
        svd_budget = 1.0 - bw

        # 生成概率分布
        if distribution == "uniform":
            svd_raw = np.ones(topk)
        elif distribution == "gaussian":
            x = np.linspace(-1, 1, topk)
            svd_raw = np.exp(-0.5 * (x / 0.5) ** 2)
        elif distribution == "dirichlet":
            svd_raw = np.random.dirichlet(alpha=[1.0] * topk)
        else:
            raise ValueError("Unsupported distribution type")

        # 归一化为剩余权重
        svd_weights = (svd_raw / svd_raw.sum() * svd_budget).round(5)

        config = {
            "base_weight": round(float(bw), 5),
            "svd_weights": svd_weights.tolist(),
            "topk": topk
        }
        configs.append((f"loss_config_{idx}", config))

    # YAML 写入（含编号注释）
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("loss_configs:\n")
        for tag, config in configs:
            f.write(f"  # {tag}\n")
            f.write("  - base_weight: {:.5f}\n".format(config["base_weight"]))
            f.write("    svd_weights: {}\n".format(config["svd_weights"]))
            f.write("    topk: {}\n".format(config["topk"]))

    print(f"✅ 成功生成 {len(configs)} 个 loss_config，保存至: {output_path}")
    return output_path


# 示例调用（你可以修改参数）
if __name__ == "__main__":
    generate_loss_configs_topk10(
        base_weight_range=(0.7, 0.9),
        num_configs=50,
        topk=10,
        distribution="dirichlet",  # 可替换为 "gaussian" 或 "uniform"
        seed=42,
        output_path="config_topk10_auto.yaml"
    )
