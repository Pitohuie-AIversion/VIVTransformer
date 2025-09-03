"""Train and evaluate the VIVTransformer with various attention mechanisms.

This script loads configuration from a YAML file and allows overriding the
location of the output results directory via command-line arguments.
"""

import argparse
import os
# 设置无头模式环境变量（必须在任何GUI相关导入之前）
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

import sys
import random
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
# 中文字体支持已通过全局 sitecustomize.py 配置，无需重复设置
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

from data.dataloader import get_loaders
from mymodels.transformer import TransformerFlowReconstructionModel
from training.trainer import train_model, test_model
from utils.visualization import plot_losses
from utils.svd10_loss import TotalLossWithSVD
from utils.config import load_config
from utils.logging_utils import setup_logging
from mymodels.components.attention_factory import ATTENTION_MODULES

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def set_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_cuda_memory_limit(fraction, device_idx=0):
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_idx)

def parse_loss_idx_from_argv():
    for i, arg in enumerate(sys.argv):
        if arg == "--loss_idx" and i+1 < len(sys.argv):
            return int(sys.argv[i+1])
    return None

def main():
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent

    parser = argparse.ArgumentParser(description="Train VIVTransformer")
    parser.add_argument(
        "-c",
        "--config",
        default=str(project_root / "modify_multi_attention" / "configs" / "config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        type=Path,
        default=project_root / "attention_results",
        help="Directory to save training results",
    )
    # 轻量化/快速验证的命令行覆盖项
    parser.add_argument("--attention-types", type=str, help="Comma-separated attention types to run, e.g. 'sge,cbam'" )
    # 新增：更细粒度/便捷的注意力控制
    parser.add_argument("--attention-type", type=str, help="Run a single attention type (overrides config and --attention-types)")
    parser.add_argument("--attention-sweep", type=str, help="Sweep multiple attention types: comma-separated list, or special values 'all'/'safe'")
    parser.add_argument("--exclude-attentions", type=str, help="Comma-separated attention types to exclude when sweeping")
    parser.add_argument("--epochs", type=int, help="Override training epochs for quick tests")
    parser.add_argument("--batch-size", type=int, help="Override data.batch_size")
    parser.add_argument("--dataset-type", type=str, choices=["auto", "pressure", "pdebench", "toy"], help="Override data.dataset_type")
    parser.add_argument("--max-samples", type=int, help="Limit max samples for quick tests (applies to supported datasets)")
    parser.add_argument("--device", type=str, help="Override device, e.g., 'cpu' or 'cuda:0'")
    parser.add_argument("--data-path", type=str, help="Override data.path to point to your dataset file (e.g., .pt or .h5)")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip loading pretrained checkpoints if present")
    # 新增：模型规模覆盖参数（用于快速降配以避免OOM）
    parser.add_argument("--num-layers", type=int, help="Override model.num_layers for quick tests")
    parser.add_argument("--d-model", dest="d_model", type=int, help="Override model.d_model for quick tests")
    parser.add_argument("--num-heads", type=int, help="Override model.num_heads for quick tests")
    parser.add_argument("--seq-len", dest="seq_len", type=int, help="Override model.seq_len for quick tests")
    parser.add_argument("--input-hw", type=str, help="Override model.input_hw as 'HxW' or 'H,W'")
    
    # 新增：输出头类型与逐token通道数覆盖
    parser.add_argument("--output-head-type", dest="output_head_type", choices=["global", "per_token"], help="Override model.output_head_type")
    parser.add_argument("--out-channels-per-token", dest="out_channels_per_token", type=int, help="Override model.out_channels_per_token")
    
    # 新增：限制运行的 loss_config 个数（从头开始取前 N 个），便于快速回归
    parser.add_argument(
        "--limit-loss-configs",
        dest="limit_loss_configs",
        type=int,
        help="Limit number of loss_configs to run from the start (e.g., 5)")

    # 新增：位置编码与时间编码的命令行覆盖
    parser.add_argument("--pe-type", dest="pe_type", choices=["learnable_1d", "sinusoidal_1d", "learnable_2d"], help="Override model.pe_type")
    parser.add_argument("--time-encoding", dest="time_encoding", choices=["embedding", "mlp"], help="Override model.time_encoding")
    # 新增：记忆融合 concat+1x1 conv 开关
    parser.add_argument("--use-memory-concat", dest="use_memory_concat", action="store_true", help="Enable memory fusion by concatenation + 1x1 conv in decoder")

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    config_path = Path(args.config)
    
    # 先设置日志，再使用log
    parent_dir = Path(args.results_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(parent_dir / "train.log")
    log = logging.getLogger(__name__)
    
    log.info("加载配置文件: %s", config_path)

    cfg = load_config(config_path)

    # 应用命令行覆盖（若提供）
    overrides = {}
    if args.attention_types:
        attn_list = [x.strip() for x in args.attention_types.split(',') if x.strip()]
        if attn_list:
            cfg["attention_types"] = attn_list
            overrides["attention_types"] = attn_list
    # 单个注意力优先级更高
    if getattr(args, "attention_type", None):
        attn = args.attention_type.strip()
        if attn:
            cfg["attention_types"] = [attn]
            overrides["attention_types"] = [attn]
    # 批量扫描/宏：在未提供单个注意力时生效
    elif getattr(args, "attention_sweep", None):
        sweep = args.attention_sweep.strip().lower()
        attn_list = []
        if sweep == "all":
            attn_list = list(ATTENTION_MODULES.keys())
        elif sweep == "safe":
            excluded_default = {"sk", "vip", "ufo", "muse", "aft"}
            attn_list = [k for k in ATTENTION_MODULES.keys() if k not in excluded_default]
        else:
            attn_list = [x.strip() for x in sweep.split(',') if x.strip()]
        if attn_list:
            cfg["attention_types"] = attn_list
            overrides["attention_types"] = attn_list
    # 排除列表：对最终 attention_types 做过滤
    if getattr(args, "exclude_attentions", None):
        excludes = {x.strip() for x in args.exclude_attentions.split(',') if x.strip()}
        if excludes:
            before = list(cfg.get("attention_types", []))
            cfg["attention_types"] = [a for a in before if a not in excludes]
            overrides["attention_types_excluded"] = sorted(list(excludes))

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
        overrides["training.epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
        overrides["data.batch_size"] = args.batch_size
    if args.dataset_type:
        cfg["data"]["dataset_type"] = args.dataset_type
        overrides["data.dataset_type"] = args.dataset_type
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
        overrides["data.max_samples"] = args.max_samples
    if args.device:
        cfg["device"] = args.device
        overrides["device"] = args.device
    if args.data_path:
        cfg["data"]["path"] = args.data_path
        overrides["data.path"] = args.data_path
    # 应用模型规模覆盖
    if getattr(args, "num_layers", None) is not None:
        cfg["model"]["num_layers"] = args.num_layers
        overrides["model.num_layers"] = args.num_layers
    if getattr(args, "d_model", None) is not None:
        cfg["model"]["d_model"] = args.d_model
        overrides["model.d_model"] = args.d_model
    if getattr(args, "num_heads", None) is not None:
        cfg["model"]["num_heads"] = args.num_heads
        overrides["model.num_heads"] = args.num_heads
    if getattr(args, "seq_len", None) is not None:
        cfg["model"]["seq_len"] = args.seq_len
        overrides["model.seq_len"] = args.seq_len
    if getattr(args, "input_hw", None):
        ihw = args.input_hw.lower().replace("x", ",").split(",")
        try:
            if len(ihw) == 2:
                h, w = int(ihw[0]), int(ihw[1])
                cfg["model"]["input_hw"] = [h, w]
                overrides["model.input_hw"] = (h, w)
        except Exception as _:
            pass

    # 新增：应用输出头与逐token通道的命令行覆盖
    if getattr(args, "output_head_type", None):
        cfg["model"]["output_head_type"] = args.output_head_type
        overrides["model.output_head_type"] = args.output_head_type
    if getattr(args, "out_channels_per_token", None) is not None:
        cfg["model"]["out_channels_per_token"] = args.out_channels_per_token
        overrides["model.out_channels_per_token"] = args.out_channels_per_token

    # 新增：应用 pe_type 与 time_encoding 的命令行覆盖
    if getattr(args, "pe_type", None):
        cfg["model"]["pe_type"] = args.pe_type
        overrides["model.pe_type"] = args.pe_type
    if getattr(args, "time_encoding", None):
        cfg["model"]["time_encoding"] = args.time_encoding
        overrides["model.time_encoding"] = args.time_encoding
    # 新增：应用 memory concat+1x1 conv 开关
    if getattr(args, "use_memory_concat", False):
        cfg["model"]["use_memory_concat"] = True
        overrides["model.use_memory_concat"] = True

    # 当选择 per_token 输出头时，自动推导并覆盖 output_dim = seq_len * C
    try:
        head = cfg["model"].get("output_head_type", "global")
        if head == "per_token":
            C = cfg["model"].get("out_channels_per_token")
            if C is None:
                log.warning("per_token 输出头已选择，但未设置 out_channels_per_token，默认使用 1")
                C = 1
                cfg["model"]["out_channels_per_token"] = C
            # 优先使用 input_hw 推导 seq_len；否则回退到显式 seq_len
            ihw = cfg["model"].get("input_hw")
            if ihw and isinstance(ihw, (list, tuple)) and len(ihw) == 2:
                seq_len_auto = int(ihw[0]) * int(ihw[1])
            else:
                seq_len_auto = cfg["model"].get("seq_len")
            if seq_len_auto is not None:
                out_dim_auto = int(seq_len_auto) * int(C)
                if cfg["model"].get("output_dim") != out_dim_auto:
                    cfg["model"]["output_dim"] = out_dim_auto
                    overrides["model.output_dim(auto)"] = out_dim_auto
            else:
                log.warning("无法自动推导 seq_len（缺少 input_hw 与 seq_len），未调整 output_dim")
    except Exception as _auto_err:
        log.debug("自动调整 output_dim 失败: %s", _auto_err)

    if overrides:
        log.info("使用命令行覆盖配置: %s", overrides)

    set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
    if "max_memory_fraction" in cfg:
        device_str = str(cfg.get("device", "cpu"))
        if "cuda" in device_str:
            if ":" in device_str:
                try:
                    device_idx = int(device_str.split(":")[-1])
                except Exception:
                    device_idx = 0
            else:
                device_idx = 0
        else:
            device_idx = 0
        try:
            set_cuda_memory_limit(cfg["max_memory_fraction"], device_idx)
        except Exception as e:
            log.warning("设置显存占比失败: %s", e)
    device = torch.device(cfg["device"]) 
    log.info("Using device: %s", device)
    log.info("Available GPUs: %s", torch.cuda.device_count())

    loss_idx = parse_loss_idx_from_argv()
    limit = getattr(args, "limit_loss_configs", None)
    if loss_idx is not None:
        log.info("只运行 loss_config_%s", loss_idx)
        loss_configs = [cfg["loss_configs"][loss_idx]]
        loss_config_ids = [f"loss_config_{loss_idx}"]
    else:
        loss_configs = cfg["loss_configs"]
        if limit is not None:
            try:
                if limit <= 0:
                    log.warning("参数 --limit-loss-configs 必须为正数，已忽略: %s", limit)
                else:
                    total = len(loss_configs)
                    limit_clamped = min(limit, total)
                    if limit_clamped < total:
                        log.info("限制 loss_configs 数量：仅运行前 %s 个（共 %s 个）", limit_clamped, total)
                    loss_configs = loss_configs[:limit_clamped]
            except Exception as _:
                pass
        loss_config_ids = [f"loss_config_{i}" for i in range(len(loss_configs))]

    ATTENTION_TYPES = cfg["attention_types"]
    vis_enabled = cfg["visualization"]["enabled"]
    failed_attention_types = []

    # 获取数据集配置参数
    dataset_type = cfg["data"].get("dataset_type", "auto")
    max_samples = cfg["data"].get("max_samples", None)
    
    train_loader, valid_loader, test_loader = get_loaders(
        cfg["data"]["path"], 
        cfg["data"]["batch_size"],
        dataset_type=dataset_type,
        max_samples=max_samples
    )

    for idx, (loss_cfg, loss_config_id) in enumerate(zip(loss_configs, loss_config_ids)):
        base_weight = loss_cfg.get("base_weight", 0.5)
        svd_weights = loss_cfg.get("svd_weights", None)
        topk = loss_cfg.get("topk", 10)

        log.info(
            "===== 当前loss设置 [%s]: base_weight=%s, svd_weights=%s, topk=%s =====",
            loss_config_id,
            base_weight,
            svd_weights,
            topk,
        )

        for attn_type in ATTENTION_TYPES:
            log.info(
                "=========== 当前测试注意力机制: %s（%s） ===========",
                attn_type,
                loss_config_id,
            )
            # 为每个 attention 建立独立日志文件（追加到根 logger），便于排查
            attention_log_handler = None
            try:
                result_dir = parent_dir / loss_config_id / attn_type
                result_dir.mkdir(parents=True, exist_ok=True)
                attention_log_handler = logging.FileHandler(result_dir / "train.log")
                attention_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                logging.getLogger().addHandler(attention_log_handler)
            except Exception as _log_err:
                log.warning("创建子日志文件失败: %s", _log_err)

            try:
                model = TransformerFlowReconstructionModel(
                    input_dim=cfg["model"]["input_dim"],
                    output_dim=cfg["model"]["output_dim"],
                    num_heads=cfg["model"]["num_heads"],
                    num_layers=cfg["model"]["num_layers"],
                    d_model=cfg["model"]["d_model"],
                    max_time_steps=cfg["model"]["max_time_steps"],
                    attention_type=attn_type,
                    seq_len=cfg["model"].get("seq_len", 32),  # 默认使用32而不是49
                    input_hw=tuple(cfg["model"].get("input_hw")) if cfg["model"].get("input_hw") else None,
                    pe_type=cfg["model"].get("pe_type", "learnable_1d"),
                    output_head_type=cfg["model"].get("output_head_type", "global"),
                    out_channels_per_token=cfg["model"].get("out_channels_per_token"),
                    time_encoding=cfg["model"].get("time_encoding", "embedding"),
                    use_memory_film=cfg["model"].get("use_memory_film", True),
                    use_memory_concat=cfg["model"].get("use_memory_concat", False),
                )

                if cfg.get("use_dataparallel", False) and torch.cuda.device_count() > 1:
                    log.info("Using DataParallel on %s GPUs!", torch.cuda.device_count())
                    model = torch.nn.DataParallel(model)
                model = model.to(device)

                criterion = TotalLossWithSVD(
                    base_weight=base_weight,
                    svd_weights=svd_weights,
                    topk=topk
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

                # 训练
                train_model(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=cfg["training"]["epochs"],
                    device=device,
                    attention_type=attn_type,
                    result_dir=result_dir,
                    cfg=cfg,
                    scheduler=None,
                    no_pretrained=args.no_pretrained,
                )
                # 测试
                test_model(
                    model=model,
                    test_loader=test_loader,
                    criterion=criterion,
                    device=device,
                    attention_type=attn_type,
                    parent_dir=result_dir,
                    cfg=cfg,
                )
                failed = False
            except Exception as e:
                failed = True
                log.error("[ERROR] 发生错误，跳过 %s (%s)", attn_type, loss_config_id)
                log.exception("[WARN] 错误详情: %s", str(e))
            finally:
                # 移除子日志 handler，避免句柄泄漏/重复写入
                try:
                    if attention_log_handler is not None:
                        logging.getLogger().removeHandler(attention_log_handler)
                        attention_log_handler.close()
                except Exception as _rm_err:
                    log.debug("移除子日志句柄异常: %s", _rm_err)
                # 保存每个 attention 的测试结果（成功或失败）
                with open(result_dir / f"test_result_{attn_type}.txt", "w") as f:
                    f.write("success\n" if not failed else "failed\n")

                # 主动释放GPU显存，避免多次循环累积/碎片化
                try:
                    if 'optimizer' in locals():
                        del optimizer
                    if 'criterion' in locals():
                        del criterion
                    if 'model' in locals():
                        del model
                    if torch.cuda.is_available() and 'cuda' in str(device):
                        torch.cuda.empty_cache()
                except Exception as _cleanup_err:
                    log.debug("清理CUDA缓存异常: %s", _cleanup_err)
    
                if not failed:
                    log.info("[OK] %s (%s) 训练完成!", attn_type, loss_config_id)
                    # 记录已完成的 attention 类型
                    completed_file = parent_dir / "completed_attention_log.txt"
                    with open(completed_file, "a") as f:
                        f.write(f"{loss_config_id}::{attn_type}\n")
                else:
                    # 记录失败的 attention 类型
                    failed_attention_types.append(f"{loss_config_id}::{attn_type}")
    
            if failed_attention_types:
                log.warning("\n[WARN] 以下loss+注意力机制训练失败，并已记录在 failed_attention_log.txt：")
                with open(parent_dir / "failed_attention_log.txt", "w") as f:
                    for item in failed_attention_types:
                        log.warning(" - %s", item)
                        f.write(item + "\n")
            else:
                log.info("\n[OK] 所有loss配置和注意力机制均运行成功！")

if __name__ == "__main__":
    main()

