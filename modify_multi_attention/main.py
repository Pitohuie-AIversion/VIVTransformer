import torch
import yaml
from data.dataloader import get_loaders
from mymodels.transformer import TransformerFlowReconstructionModel
from training.trainer import train_model, test_model
from utils.visualization import plot_losses
import os
import matplotlib.pyplot as plt  # 明确导入matplotlib
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式，防止弹窗

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ATTENTION_TYPES = [
#     # "external", "self", "simplified_self", "muse", "ufo", "aft", "vip", "halo",
#     "se", "sk", "cbam", "bam", "eca", "danet", "psa", "shuffle", "muse", "sge", "a2", "aft",
#     "outlook", "vip", "coatnet", "halo", "polarized", "cot",
#     "residual", "s2", "crossformer", "moa", "dat", "parnet", "mobilevit", "mobilevitv2"


import torch
import yaml
from data.dataloader import get_loaders
from mymodels.transformer import TransformerFlowReconstructionModel
from training.trainer import train_model, test_model
from utils.visualization import plot_losses
import os
import matplotlib.pyplot as plt  # 明确导入matplotlib
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式，防止弹窗

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # 强制OpenMP只用单线程
os.environ["MKL_NUM_THREADS"] = "1"

from utils.losses import CustomLossWithMask, StandardMSELoss, SimpleLossWithMask
from utils.mask_utils import generate_box_mask
# main.py 相关片段
from utils.losses import CustomLossWithMask, StandardMSELoss, SVDMainModeLoss, TotalLossWithSVD

def get_loss_function(cfg):
    base_loss = None
    if cfg['training']['loss_function'] == 'custom_masked':
        base_loss = CustomLossWithMask(
            lambda_l2=cfg['training'].get('lambda_l2', 1.0),
            lambda_mask=cfg['training'].get('lambda_mask', 0.1)
        )
    else:
        base_loss = StandardMSELoss()

    # 判断是否开启 SVD主模态loss
    if cfg['training'].get('svd_main_mode_enabled', False):
        svd_loss = SVDMainModeLoss(
            h=cfg['model'].get('output_h', 200),
            w=cfg['model'].get('output_w', 200),
            mode=cfg['training'].get('main_mode', 'mse')
        )
        return TotalLossWithSVD(
            base_loss,
            svd_loss,
            lambda_base=cfg['training'].get('lambda_base', 1.0),
            lambda_main=cfg['training'].get('lambda_main', 0.1)
        )
    else:
        return base_loss

def main():
    with open('modify_multi_attention/configs/config.yaml', 'r', encoding="utf-8") as f:

        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, input_dim, output_dim = get_loaders(cfg["data"]["path"],
                                                                                 cfg["data"]["batch_size"])

    ATTENTION_TYPES = cfg["attention_types"]
    vis_enabled = cfg["visualization"]["enabled"]
    parent_dir = "attention_results"
    os.makedirs(parent_dir, exist_ok=True)

    failed_attention_types = []

    for attn_type in ATTENTION_TYPES:
        print(f"\n=========== 当前测试注意力机制: {attn_type} ===========")

        try:
            model = TransformerFlowReconstructionModel(
                input_dim=input_dim,
                output_dim=output_dim,
                num_heads=cfg["model"]["num_heads"],
                num_layers=cfg["model"]["num_layers"],
                d_model=cfg["model"]["d_model"],
                max_time_steps=cfg["model"]["max_time_steps"],
                attention_type=attn_type
            ).to(device)

            # 选择损失函数
            criterion = get_loss_function(cfg)

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

            # 如果使用掩码，准备mask_boxes供训练函数使用
            use_mask = cfg["training"].get("mask_enabled", False)
            mask_boxes = cfg["training"].get("mask_regions", []) if use_mask else None

            trained_model, train_loss, valid_loss, test_loss = train_model(
                model, train_loader, valid_loader, test_loader,
                criterion, optimizer, cfg["training"]["epochs"],
                device, cfg["training"]["early_stop_patience"],
                attention_type=attn_type,
                use_mask=use_mask,
                mask_boxes=mask_boxes
            )

            result_dir = os.path.join(parent_dir, attn_type)
            os.makedirs(result_dir, exist_ok=True)

            best_model_path = os.path.join(result_dir, f"best_model_{attn_type}.pt")
            torch.save(trained_model.state_dict(), best_model_path)

            if vis_enabled:
                # 保存损失曲线图像
                plot_losses(train_loss, valid_loss, test_loss)
                loss_fig_path = os.path.join(result_dir, f"loss_curve_{attn_type}.png")
                plt.savefig(loss_fig_path)
                plt.close()

            final_test_loss = test_model(
                trained_model, test_loader, criterion, device,
                attention_type=attn_type,
                parent_dir="attention_results"
            )

            # 保存最终测试结果
            with open(os.path.join(result_dir, f"test_result_{attn_type}.txt"), 'w') as f:
                f.write(f"Test Loss for {attn_type}: {final_test_loss}\n")

            print(f"✅ {attn_type} 训练完成！")

        except Exception as e:
            print(f"❌ 发生错误，跳过 {attn_type} 注意力机制")
            print(f"⚠️ 错误详情: {str(e)}")
            failed_attention_types.append(attn_type)

    if failed_attention_types:
        with open(os.path.join(parent_dir, "failed_attention_log.txt"), "w") as f:
            for attn in failed_attention_types:
                f.write(f"{attn}\n")
        print("\n⚠️ 以下注意力机制训练失败，并已记录：")
        print("\n".join(failed_attention_types))
    else:
        print("\n🎉 所有注意力机制均运行成功！")

if __name__ == "__main__":
    main()  # 启动主函数
