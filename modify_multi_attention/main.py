"""Train and evaluate the VIVTransformer with various attention mechanisms.

This script loads configuration from a YAML file and allows overriding the
location of the output results directory via command-line arguments.
"""

import argparse
import os
import sys
import random
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from data.dataloader import get_loaders, get_pdebench_loaders
from mymodels.transformer import TransformerFlowReconstructionModel
from training.trainer import train_model, test_model
from utils.visualization import plot_losses
from utils.svd10_loss import TotalLossWithSVD
from utils.config import load_config
from utils.logging_utils import setup_logging
from utils.dimension_utils import (
    validate_model_dimensions, 
    auto_update_model_dimensions,
    create_dimension_summary
)

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
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    config_path = Path(args.config)
    
    # Setup logging first
    parent_dir = Path(args.results_dir)
    parent_dir.mkdir(exist_ok=True)
    setup_logging(parent_dir / "train.log")
    log = logging.getLogger(__name__)
    
    log.info("鍔犺浇閰嶇疆鏂囦欢: %s", config_path)

    cfg = load_config(config_path)
    
    # 验证和自动修正数据维度配置
    log.info("验证数据维度配置...")
    if not validate_model_dimensions(cfg):
        log.warning("检测到维度配置不匹配，自动修正中...")
        cfg = auto_update_model_dimensions(cfg)
        log.info("维度配置已自动修正")
    
    # 鎵撳嵃缁村害鎽樿淇℃伅
    log.info("\n%s", create_dimension_summary(cfg))

    set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
    if "max_memory_fraction" in cfg:
        device_idx = int(str(cfg["device"]).split(":")[-1])
        set_cuda_memory_limit(cfg["max_memory_fraction"], device_idx)
    device = torch.device(cfg["device"])
    log.info("Using device: %s", device)
    log.info("Available GPUs: %s", torch.cuda.device_count())

    loss_idx = parse_loss_idx_from_argv()
    if loss_idx is not None:
        log.info("鍙繍琛?loss_config_%s", loss_idx)
        loss_configs = [cfg["loss_configs"][loss_idx]]
        loss_config_ids = [f"loss_config_{loss_idx}"]
    else:
        loss_configs = cfg["loss_configs"]
        loss_config_ids = [f"loss_config_{i}" for i in range(len(loss_configs))]

    ATTENTION_TYPES = cfg["attention_types"]
    vis_enabled = cfg["visualization"]["enabled"]
    failed_attention_types = []

    # Load data based on dataset type
    dataset_type = cfg["data"].get("dataset_type", "viv")
    
    if dataset_type == "pdebench":
        pdebench_cfg = cfg["data"]["pdebench"]
        
        # Check if using processed data
        use_processed_data = pdebench_cfg.get("use_processed_data", False)
        
        if use_processed_data:
            # Use processed data path
            data_path = pdebench_cfg.get("processed_data_path", "pdebench_processed_center_crop.pt")
            log.info("浣跨敤棰勫鐞嗘暟鎹? %s", data_path)
        else:
            # Use original HDF5 data path
            data_path = cfg["data"]["path"]
            log.info("浣跨敤鍘熷HDF5鏁版嵁: %s", data_path)
        
        train_loader, valid_loader, test_loader, dataset_info = get_pdebench_loaders(
            data_path,
            cfg["data"]["batch_size"],
            input_key=pdebench_cfg.get("input_key", "nu"),
            output_key=pdebench_cfg.get("output_key", "tensor"),
            normalize=pdebench_cfg.get("normalize", True),
            flatten=pdebench_cfg.get("flatten", True),
            simulate_time_series=pdebench_cfg.get("simulate_time_series", False),
            time_steps_per_sample=pdebench_cfg.get("time_steps_per_sample", 10),
            max_time_value=pdebench_cfg.get("max_time_value", 1.0),
            use_processed_data=use_processed_data
        )
        
        log.info("PDEBench鏁版嵁鍔犺浇瀹屾垚")
        if use_processed_data:
            log.info("  璁粌闆嗘牱鏈暟: %d", len(dataset_info['train_dataset']))
            log.info("  楠岃瘉闆嗘牱鏈暟: %d", len(dataset_info['valid_dataset']))
            log.info("  娴嬭瘯闆嗘牱鏈暟: %d", len(dataset_info['test_dataset']))
        else:
            log.info("  鎬绘牱鏈暟: %d", len(dataset_info['dataset']))
    
    else:
        # Original VIV data loading
        train_loader, valid_loader, test_loader = get_loaders(
            cfg["data"]["path"],
            cfg["data"]["batch_size"]
        )
        log.info("VIV鏁版嵁鍔犺浇瀹屾垚")

    for idx, (loss_cfg, loss_config_id) in enumerate(zip(loss_configs, loss_config_ids)):
        base_weight = loss_cfg.get("base_weight", 0.5)
        svd_weights = loss_cfg.get("svd_weights", None)
        topk = loss_cfg.get("topk", 10)

        log.info(
            "===== 褰撳墠loss璁剧疆 [%s]: base_weight=%s, svd_weights=%s, topk=%s =====",
            loss_config_id,
            base_weight,
            svd_weights,
            topk,
        )

        for attn_type in ATTENTION_TYPES:
            log.info(
                "=========== 褰撳墠娴嬭瘯娉ㄦ剰鍔涙満鍒? %s锛?s锛?===========",
                attn_type,
                loss_config_id,
            )
            try:
                result_dir = parent_dir / loss_config_id / attn_type
                result_dir.mkdir(parents=True, exist_ok=True)

                model = TransformerFlowReconstructionModel(
                    input_dim=cfg["model"]["input_dim"],
                    output_dim=cfg["model"]["output_dim"],
                    num_heads=cfg["model"]["num_heads"],
                    num_layers=cfg["model"]["num_layers"],
                    d_model=cfg["model"]["d_model"],
                    max_time_steps=cfg["model"]["max_time_steps"],
                    attention_type=attn_type,
                    seq_len=cfg["model"].get("seq_len", 49)
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

                trained_model, train_loss, valid_loss, test_loss = train_model(
                    model, train_loader, valid_loader, test_loader,
                    criterion, optimizer, cfg["training"]["epochs"],
                    device, cfg["training"]["early_stop_patience"],
                    attention_type=attn_type,
                    result_dir=result_dir,
                    cfg=cfg
                )

                best_model_path = result_dir / f"best_model_{attn_type}.pt"
                torch.save(trained_model.state_dict(), best_model_path)

                if vis_enabled:
                    plot_losses(train_loss, valid_loss, test_loss)
                    loss_fig_path = result_dir / f"loss_curve_{attn_type}.png"
                    plt.savefig(loss_fig_path)
                    plt.close()

                final_test_loss = test_model(
                    trained_model, test_loader, criterion, device,
                    attention_type=attn_type,
                    parent_dir=result_dir,
                    cfg=cfg
                )

                test_result_file = result_dir / f"test_result_{attn_type}.txt"
                with open(test_result_file, "w") as f:
                    f.write(f"Test Loss for {attn_type}: {final_test_loss}\n")

                log.info("✅ %s (%s) 训练完成！", attn_type, loss_config_id)

            except Exception as e:
                log.error("❌ 发生错误，跳过 %s (%s)", attn_type, loss_config_id)
                log.exception("⚠️ 错误详情: %s", str(e))
                failed_attention_types.append(f"{loss_config_id}::{attn_type}")

    if failed_attention_types:
        with open(parent_dir / "failed_attention_log.txt", "w") as f:
            for info in failed_attention_types:
                f.write(f"{info}\n")
        log.warning("\n⚠️ 以下loss+注意力机制训练失败，并已记录在 failed_attention_log.txt：")
        for info in failed_attention_types:
            log.warning(info)
    else:
        log.info("\n🎉 所有loss配置和注意力机制均运行成功！")

if __name__ == "__main__":
    main()
