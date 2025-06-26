import torch
import yaml
from data.dataloader import get_loaders
from mymodels.transformer import TransformerFlowReconstructionModel
from training.trainer import train_model, test_model
from utils.visualization import plot_losses
# import os
# import matplotlib.pyplot as plt  # 明确导入matplotlib
# import matplotlib
# matplotlib.use('Agg')  # 使用非交互模式，防止弹窗
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ATTENTION_TYPES = [
#     # "external", "self", "simplified_self", "muse", "ufo", "aft", "vip", "halo",
#     "se", "sk", "cbam", "bam", "eca", "danet", "psa", "shuffle", "muse", "sge", "a2", "aft",
#     "outlook", "vip", "coatnet", "halo", "polarized", "cot",
#     "residual", "s2", "crossformer", "moa", "dat", "parnet", "mobilevit", "mobilevitv2"
# ]
import torch
import yaml
import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from modify_multi_attention.data.dataloader import get_loaders
from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
from modify_multi_attention.training.trainer import train_model, test_model
from modify_multi_attention.utils.visualization import plot_losses
from modify_multi_attention.utils.svd10_loss import TotalLossWithSVD

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
    this_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(this_file))
    config_path = os.path.join(project_root, 'modify_multi_attention', 'configs', 'config.yaml')
    print(f"加载配置文件: {config_path}")

    with open(config_path, 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
    if "max_memory_fraction" in cfg:
        device_idx = int(str(cfg["device"]).split(":")[-1])
        set_cuda_memory_limit(cfg["max_memory_fraction"], device_idx)
    device = torch.device(cfg["device"])
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    loss_idx = parse_loss_idx_from_argv()
    if loss_idx is not None:
        print(f"只运行 loss_config_{loss_idx}")
        loss_configs = [cfg["loss_configs"][loss_idx]]
        loss_config_ids = [f"loss_config_{loss_idx}"]
    else:
        loss_configs = cfg["loss_configs"]
        loss_config_ids = [f"loss_config_{i}" for i in range(len(loss_configs))]

    ATTENTION_TYPES = cfg["attention_types"]
    vis_enabled = cfg["visualization"]["enabled"]
    parent_dir = "attention_results"
    os.makedirs(parent_dir, exist_ok=True)
    failed_attention_types = []

    train_loader, valid_loader, test_loader = get_loaders(
        cfg["data"]["path"],
        cfg["data"]["batch_size"]
    )

    for idx, (loss_cfg, loss_config_id) in enumerate(zip(loss_configs, loss_config_ids)):
        base_weight = loss_cfg.get("base_weight", 0.5)
        svd_weights = loss_cfg.get("svd_weights", None)
        topk = loss_cfg.get("topk", 10)

        print(f"\n===== 当前loss设置 [{loss_config_id}]: base_weight={base_weight}, svd_weights={svd_weights}, topk={topk} =====")

        for attn_type in ATTENTION_TYPES:
            print(f"\n=========== 当前测试注意力机制: {attn_type}（{loss_config_id}） ===========")
            try:
                result_dir = os.path.join(parent_dir, loss_config_id, attn_type)
                os.makedirs(result_dir, exist_ok=True)

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
                    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs!")
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

                best_model_path = os.path.join(result_dir, f"best_model_{attn_type}.pt")
                torch.save(trained_model.state_dict(), best_model_path)

                if vis_enabled:
                    plot_losses(train_loss, valid_loss, test_loss)
                    loss_fig_path = os.path.join(result_dir, f"loss_curve_{attn_type}.png")
                    plt.savefig(loss_fig_path)
                    plt.close()

                final_test_loss = test_model(
                    trained_model, test_loader, criterion, device,
                    attention_type=attn_type,
                    parent_dir=result_dir,
                    cfg=cfg
                )

                test_result_file = os.path.join(result_dir, f"test_result_{attn_type}.txt")
                with open(test_result_file, 'w') as f:
                    f.write(f"Test Loss for {attn_type}: {final_test_loss}\n")

                print(f"✅ {attn_type} ({loss_config_id}) 训练完成！")

            except Exception as e:
                print(f"❌ 发生错误，跳过 {attn_type} ({loss_config_id})")
                print(f"⚠️ 错误详情: {str(e)}")
                failed_attention_types.append(f"{loss_config_id}::{attn_type}")

    if failed_attention_types:
        with open(os.path.join(parent_dir, "failed_attention_log.txt"), "w") as f:
            for info in failed_attention_types:
                f.write(f"{info}\n")
        print(f"\n⚠️ 以下loss+注意力机制训练失败，并已记录在 failed_attention_log.txt：")
        print("\n".join(failed_attention_types))
    else:
        print("\n🎉 所有loss配置和注意力机制均运行成功！")

if __name__ == "__main__":
    main()
