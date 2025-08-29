#!/usr/bin/env python3
"""
单注意力机制的极端loss配置并行扫描脚本
直接调用 modify_multi_attention/main.py --attention-type [NAME] --loss_idx [i]
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import multiprocessing as mp
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
    gpu_id, loss_idx, attention_type, project_root, epochs = args
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTHONUNBUFFERED'] = '1'
    
    # 构建命令
    trainer_path = Path(project_root) / "modify_multi_attention" / "main.py"
    # 尝试使用临时配置文件（若存在）
    temp_config = Path(project_root) / "temp_extreme_loss_config.yaml"
    cmd = [
        sys.executable, str(trainer_path),
        "--attention-type", attention_type,
        "--loss_idx", str(loss_idx),
        "--epochs", str(epochs),
        "--device", "cuda",
        "--dataset-type", "toy"
    ]
    if temp_config.exists():
        cmd += ["--config", str(temp_config)]
    
    # 输出目录
    results_dir = Path(project_root) / "attention_results" / f"loss_config_{loss_idx}" / attention_type
    results_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = results_dir / f"train_{loss_idx}_{attention_type}.log"
    
    print(f"[GPU {gpu_id}] 启动: {attention_type} (loss_config_{loss_idx})")
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.run(
                cmd, 
                env=env, 
                cwd=project_root,
                stdout=f, 
                stderr=subprocess.STDOUT,
                timeout=3600  # 1小时超时
            )
        return loss_idx, process.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[GPU {gpu_id}] 超时: {attention_type} (loss_config_{loss_idx})")
        return loss_idx, False
    except Exception as e:
        print(f"[GPU {gpu_id}] 错误: {attention_type} (loss_config_{loss_idx}) - {e}")
        return loss_idx, False

def create_temp_config_with_extreme_losses(project_root, num_configs=10):
    """创建包含极端 loss 配置的临时配置文件"""
    import yaml
    
    # 截取指定数量的配置
    configs = EXTREME_LOSS_CONFIGS[:num_configs]
    
    # 基础配置模板
    base_config = {
        "model": {
            "input_dim": 3,
            "output_dim": 3,
            "num_heads": 8,
            "num_layers": 6,
            "d_model": 512,
            "max_time_steps": 32,
            "seq_len": 32
        },
        "training": {
            "epochs": 1000,
            "learning_rate": 0.001,
            "early_stopping": {"patience": 50, "min_delta": 1e-6}
        },
        "data": {
            "path": str(Path(project_root) / "generate_data" / "preprocessed_data"),
            "batch_size": 16,
            "dataset_type": "toy"
        },
        "device": "cuda",
        "loss_configs": configs
    }
    
    temp_config_path = Path(project_root) / "temp_extreme_loss_config.yaml"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 创建临时配置: {temp_config_path} (包含 {len(configs)} 个极端组合)")
    return temp_config_path

def plot_loss_curves(project_root, attention_type, num_configs):
    """绘制该注意力类型下不同 loss 配置的对比曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(16, 10))
    
    collected = {}
    for i in range(num_configs):
        loss_log = Path(project_root) / "attention_results" / f"loss_config_{i}" / attention_type / "loss_logs" / "loss_log.txt"
        if not loss_log.exists():
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
    parser.add_argument("-G", "--gpus", default="0,1", help="GPU列表 (逗号分隔)")
    parser.add_argument("-M", "--max-per-gpu", type=int, default=2, help="每GPU最大并行任务数")
    parser.add_argument("-P", "--project-root", help="项目根目录")
    
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
    
    # 解析GPU列表
    gpu_list = [int(g.strip()) for g in args.gpus.split(',') if g.strip().isdigit()]
    print(f"🎯 GPU列表: {gpu_list}, 每GPU最大并行: {args.max_per_gpu}")
    
    # 创建临时配置文件
    temp_config = create_temp_config_with_extreme_losses(project_root, args.num_configs)
    
    try:
        # 准备任务队列
        tasks = []
        for i in range(args.num_configs):
            # 轮流分配GPU
            gpu_id = gpu_list[i % len(gpu_list)]
            tasks.append((gpu_id, i, args.attention, str(project_root), args.epochs))
        
        # 使用进程池并行执行
        max_workers = len(gpu_list) * args.max_per_gpu
        print(f"🔄 启动并行训练 (最大工作进程: {max_workers})")
        
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = executor.map(run_single_loss_config, tasks)
            results = list(futures)
        
        # 统计结果
        successful = sum(1 for _, success in results if success)
        total_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"📊 成功: {successful}/{len(results)} 个配置")
        print(f"⏱️  总耗时: {total_time/60:.1f} 分钟")
        
        # 绘制对比图
        print(f"\n📈 生成对比图...")
        plot_loss_curves(project_root, args.attention, args.num_configs)
        
    finally:
        # 清理临时文件
        if temp_config.exists():
            temp_config.unlink()
            print(f"🗑️  已清理临时配置: {temp_config}")

if __name__ == "__main__":
    main()