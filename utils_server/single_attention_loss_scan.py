#!/usr/bin/env python3
"""
单注意力机制的极端loss配置并行扫描脚本
改为调用 generate_data/dynamic_resolution_trainer.py --attention-type [NAME] --loss_idx [i]
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# 极端配置组合 (topk=3, 专注前三模态, 语义去重)
EXTREME_LOSS_CONFIGS = [
    {"base_weight": 1.0, "topk": 3, "svd_weights": [0.0, 0.0, 0.0]},          # 仅 base
    {"base_weight": 0.0, "topk": 3, "svd_weights": [1.0, 1.0, 1.0]},          # 仅 SVD（等权）
    {"base_weight": 0.0, "topk": 3, "svd_weights": [1.0, 0.0, 0.0]},          # 纯第1模态
    {"base_weight": 0.0, "topk": 3, "svd_weights": [0.0, 1.0, 0.0]},          # 纯第2模态
    {"base_weight": 0.0, "topk": 3, "svd_weights": [0.0, 0.0, 1.0]},          # 纯第3模态
    {"base_weight": 0.1, "topk": 3, "svd_weights": [0.8, 0.15, 0.05]},        # 金字塔 1>2>3
    {"base_weight": 0.1, "topk": 3, "svd_weights": [0.05, 0.15, 0.8]},        # 反向金字塔 3>2>1
    {"base_weight": 0.5, "topk": 3, "svd_weights": [1.0, 1.0, 1.0]},          # base+SVD 等权
    {"base_weight": 0.2, "topk": 3, "svd_weights": [1.0, 1.0, 1.0]},          # base 较弱
    {"base_weight": 0.3, "topk": 3, "svd_weights": [0.6, 0.3, 0.1]},          # 递减 1>2>3（另一组）
]


def run_single_loss_config(args):
    """在指定 GPU 上运行单个 loss 配置"""
    # 增加超时参数（秒），0 表示不设超时
    gpu_id, loss_idx, attention_type, project_root, epochs, ds_type, data_path, batch_size, max_samples, run_id, timeout_seconds = args

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTHONUNBUFFERED'] = '1'

    # 目标训练脚本（动态分辨率训练器）
    trainer_path = Path(project_root) / "generate_data" / "dynamic_resolution_trainer.py"
    # 尝试使用临时配置文件（若存在）
    temp_config = Path(project_root) / "temp_extreme_loss_config.yaml"

    # 输出目录（也作为 dynamic 的 --results-root），引入 run_id 子目录避免覆盖
    results_dir = Path(project_root) / "attention_results" / f"loss_config_{loss_idx}" / attention_type / f"run_{run_id}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = [
        sys.executable, str(trainer_path),
        "--attention-type", attention_type,
        "--loss_idx", str(loss_idx),
        "--epochs", str(epochs),
        "--device", "cuda",
        "--results-root", str(results_dir),
    ]
    # 数据相关覆盖（按需附加）
    if data_path:
        cmd += ["--data-path", data_path]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if max_samples is not None:
        # dynamic_resolution_trainer.py 使用 --num-samples
        cmd += ["--num-samples", str(max_samples)]
    if temp_config.exists():
        cmd += ["--config", str(temp_config)]

    log_file = results_dir / f"train_{loss_idx}_{attention_type}.log"

    print(f"[GPU {gpu_id}] 启动: {attention_type} (loss_config_{loss_idx}, run_id={run_id})")

    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            if timeout_seconds and timeout_seconds > 0:
                process = subprocess.run(
                    cmd,
                    env=env,
                    cwd=str(project_root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds
                )
            else:
                process = subprocess.run(
                    cmd,
                    env=env,
                    cwd=str(project_root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
        return loss_idx, process.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] 超时: {attention_type} (loss_config_{loss_idx}, run_id={run_id})")
        return loss_idx, False
    except Exception as e:
        print(f"[GPU {gpu_id}] 错误: {attention_type} (loss_config_{loss_idx}, run_id={run_id}) - {e}")
        return loss_idx, False


def create_temp_config_with_extreme_losses(project_root, num_configs=10, dataset_type=None, data_path=None, batch_size=None, max_samples=None):
    """创建包含极端 loss 配置的临时配置文件（适配动态分辨率训练器）"""
    import yaml

    # 截取指定数量的配置
    configs = EXTREME_LOSS_CONFIGS[:num_configs]

    base_config = None

    # 优先使用单注意力扫描专用配置作为基底（如存在），否则回退到通用 dynamic_config.yaml
    server_scan_cfg = Path(project_root) / "generate_data" / "dynamic_config_server_single_attention_loss_scan.yaml"
    base_cfg_path = server_scan_cfg if server_scan_cfg.exists() else Path(project_root) / "generate_data" / "dynamic_config.yaml"
    if base_cfg_path.exists():
        with open(base_cfg_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)

    # 兜底的基础配置（与 dynamic_resolution_trainer.py 兼容的最小集合）
    if base_config is None:
        base_config = {
            "model": {
                "num_heads": 8,
                "num_layers": 4,
                "d_model": 256,
                # input_dim/output_dim 会在训练器内根据分辨率或SVD自动计算
            },
            "training": {
                "epochs": 200,
                "learning_rate": 0.001,
                "early_stopping": {"patience": 20, "min_delta": 1e-6}
            },
            "data": {
                # 提供默认分辨率，用户可用 CLI 覆盖
                "input_resolution": [32, 32],
                "output_resolution": [128, 128],
                "path": str(Path(project_root) / "generate_data" / "preprocessed_data"),
                "batch_size": 16,
                # SVD 配置节点需要存在，训练器会根据 CLI 覆盖/自动处理
                "svd_projection": {"enabled": False, "n_modes": 128}
            },
            "visualization": {"enabled": True},
            "device": "cuda",
            "loss_configs": configs
        }

    # 覆盖必要字段
    base_config["loss_configs"] = configs
    base_config.setdefault("data", {})
    # 输入可选覆盖
    if data_path:
        base_config["data"]["path"] = data_path
    if batch_size:
        base_config["data"]["batch_size"] = batch_size
    if max_samples is not None:
        # dynamic 使用 num_samples 字段
        base_config.setdefault("data", {})
        base_config["data"]["num_samples"] = int(max_samples)
    # 确保分辨率字段存在（若基底缺失）
    base_config["data"].setdefault("input_resolution", [32, 32])
    base_config["data"].setdefault("output_resolution", [128, 128])
    # 设备
    base_config["device"] = "cuda"

    temp_config_path = Path(project_root) / "temp_extreme_loss_config.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ 创建临时配置: {temp_config_path} (包含 {len(configs)} 个极端组合)")
    return temp_config_path


def plot_loss_curves(project_root, attention_type, num_configs, run_id=None):
    """绘制该注意力类型下不同 loss 配置的对比曲线，支持按 run_id 或自动选择最新 run_ 目录"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 10))

    collected = {}
    for i in range(num_configs):
        # 优先使用提供的 run_id；否则自动选择该配置下最新的 run_* 目录；如均不存在，回退到历史的无 run 目录
        loss_log = None
        if run_id:
            base_dir = Path(project_root) / "attention_results" / f"loss_config_{i}" / attention_type / f"run_{run_id}"
            candidate = base_dir / "loss_logs" / "loss_log.txt"
            if candidate.exists():
                loss_log = candidate
            else:
                # 如果指定 run_id 不存在，尝试最新 run_ 目录
                base_dir_parent = Path(project_root) / "attention_results" / f"loss_config_{i}" / attention_type
                run_dirs = sorted([d for d in base_dir_parent.glob("run_*") if d.is_dir()], reverse=True)
                if run_dirs:
                    candidate = run_dirs[0] / "loss_logs" / "loss_log.txt"
                    if candidate.exists():
                        loss_log = candidate
        else:
            base_dir_parent = Path(project_root) / "attention_results" / f"loss_config_{i}" / attention_type
            run_dirs = sorted([d for d in base_dir_parent.glob("run_*") if d.is_dir()], reverse=True)
            if run_dirs:
                candidate = run_dirs[0] / "loss_logs" / "loss_log.txt"
                if candidate.exists():
                    loss_log = candidate
            else:
                candidate = base_dir_parent / "loss_logs" / "loss_log.txt"
                if candidate.exists():
                    loss_log = candidate

        if not loss_log or not loss_log.exists():
            continue

        try:
            epochs, valid_losses = [], []
            with open(loss_log, 'r', encoding='utf-8') as f:
                next(f)  # 跳过标题行
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3 and parts[0].isdigit():
                        epochs.append(int(parts[0]))
                        valid_losses.append(float(parts[2]))

            if epochs and valid_losses:
                config_desc = f"Config_{i}: {EXTREME_LOSS_CONFIGS[i]}"
                plt.plot(epochs, valid_losses, label=config_desc, linewidth=2)
                collected[f"loss_config_{i}"] = (epochs, valid_losses)

        except Exception as e:
            print(f"跳过 loss_config_{i}: {e}")

    if not collected:
        print("⚠️ 未收集到任何有效的 loss 曲线")
        return

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss (log10 scale)', fontsize=12)
    plt.yscale('log', base=10)
    plt.title(f'{attention_type.upper()} Attention: Extreme Loss Config Comparison', fontsize=14)
    plt.grid(True, which='both', ls='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    # 输出文件名携带 run_id 以避免覆盖
    if run_id:
        output_path = Path(project_root) / f"{attention_type}_extreme_loss_comparison_{run_id}.png"
    else:
        output_path = Path(project_root) / f"{attention_type}_extreme_loss_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 已保存对比图: {output_path}")
    print(f"📈 成功收集 {len(collected)} 条曲线")


def main():
    parser = argparse.ArgumentParser(description="单注意力机制的极端loss配置并行扫描")
    parser.add_argument("-A", "--attention", required=True, help="注意力类型 (如 eca, s2, bam)")
    parser.add_argument("-N", "--num-configs", type=int, default=10, help="使用的极端配置数量")
    parser.add_argument("-E", "--epochs", type=int, default=1000, help="训练轮次")
    parser.add_argument("-G", "--gpus", default="0", help="GPU列表 (逗号分隔)")
    parser.add_argument("-M", "--max-per-gpu", type=int, default=1, help="每GPU最大并行任务数")
    parser.add_argument("-P", "--project-root", help="项目根目录")
    # 数据集相关可选参数（直通 dynamic_resolution_trainer.py）
    parser.add_argument("--dataset-type", choices=["auto", "pressure", "pdebench", "toy"], help="数据集类型覆盖（动态训练器不使用，保留兼容）")
    parser.add_argument("--data-path", type=str, help="数据文件/目录路径覆盖（如 .h5/.hdf5）")
    parser.add_argument("--batch-size", type=int, help="批大小覆盖")
    parser.add_argument("--max-samples", type=int, help="限制样本数量（快速验证，将映射到 --num-samples）")
    # 新增：运行ID、只重绘、跳过已存在、超时
    parser.add_argument("--run-id", type=str, help="运行ID（默认时间戳），用于避免覆盖并支持复现/续跑")
    parser.add_argument("--replot-only", action="store_true", help="仅根据已有结果重新绘图，不进行训练")
    parser.add_argument("--skip-existing", action="store_true", help="已存在loss日志则跳过该配置的训练")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="单个训练进程超时(秒)，0表示不设超时，适合1000+epoch长跑")

    args = parser.parse_args()

    # 项目根目录
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        project_root = Path(__file__).parent.parent.resolve()

    print(f"🚀 单注意力机制极端loss配置扫描")
    print(f"📁 项目根目录: {project_root}")
    print(f"🧠 注意力类型: {args.attention}")
    print(f"🔢 配置数量: {args.num_configs}")
    print(f"⚙️  训练轮次: {args.epochs}")
    if args.dataset_type:
        print(f"🗂️  数据集类型(ignored by dynamic): {args.dataset_type}")
    if args.data_path:
        print(f"📄 数据路径: {args.data_path}")
    if args.timeout_seconds:
        print(f"⏱️  单任务超时: {args.timeout_seconds} s (0表示不设超时)")

    # 生成/读取 run_id
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    print(f"🧾 运行ID: {run_id}")

    # 仅重绘模式：直接绘图并退出
    if args.replot_only:
        print("🖼️ 仅重新绘图（不训练）")
        plot_loss_curves(project_root, args.attention, args.num_configs, run_id)
        return

    # 解析GPU列表
    gpu_list = [int(g.strip()) for g in args.gpus.split(',') if g.strip().isdigit()]
    print(f"🎯 GPU列表: {gpu_list}, 每GPU最大并行: {args.max_per_gpu}")

    # 创建临时配置文件（包含 loss_configs，供 dynamic 按 loss_idx 选择）
    temp_config = create_temp_config_with_extreme_losses(
        project_root,
        args.num_configs,
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )

    try:
        # 准备任务队列（支持 --skip-existing）
        tasks = []
        skipped = 0
        for i in range(args.num_configs):
            # 轮流分配GPU
            gpu_id = gpu_list[i % len(gpu_list)]
            # 如已存在该配置的 loss_log，则可跳过
            if args.skip_existing:
                expected_log = Path(project_root) / "attention_results" / f"loss_config_{i}" / args.attention / f"run_{run_id}" / "loss_logs" / "loss_log.txt"
                if expected_log.exists():
                    print(f"⏭️ 跳过 loss_config_{i}（已存在 loss 日志）：{expected_log}")
                    skipped += 1
                    continue
            tasks.append((gpu_id, i, args.attention, str(project_root), args.epochs, args.dataset_type, args.data_path, args.batch_size, args.max_samples, run_id, args.timeout_seconds))

        if not tasks:
            print("✅ 无需训练（全部已存在或被跳过），直接重绘图...")
            plot_loss_curves(project_root, args.attention, args.num_configs, run_id)
            return

        # 使用进程池并行执行
        max_workers = max(1, len(gpu_list) * args.max_per_gpu)
        print(f"🔄 启动并行训练 (最大工作进程: {max_workers})")

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_single_loss_config, tasks))

        # 统计结果
        successful = sum(1 for _, success in results if success)
        total_time = time.time() - start_time

        print(f"\n✅ 训练完成!")
        print(f"📊 成功: {successful}/{len(results)} 个配置（已跳过 {skipped} 个）")
        print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")

        # 绘制对比图（按 run_id）
        print(f"\n📈 生成对比图...")
        plot_loss_curves(project_root, args.attention, args.num_configs, run_id)

    finally:
        # 清理临时文件
        if 'temp_config' in locals() and temp_config and temp_config.exists():
            temp_config.unlink()
            print(f"🗑️  已清理临时配置: {temp_config}")


if __name__ == "__main__":
    main()