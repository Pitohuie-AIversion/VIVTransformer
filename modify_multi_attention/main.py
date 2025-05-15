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
# ]

from utils.losses import (
    CustomLossWithMask,
    StandardMSELoss,
    SimpleLossWithMask,
    MultiModeWeightedMSELoss
)
# from utils.mask_utils import generate_box_mask  # 如需单盒掩码可启用

def get_loss_function(cfg):
    """
    动态选择损失函数
    :param cfg: 配置文件
    :return: 损失函数
    """
    loss_name = cfg['training']['loss_function']
    if loss_name == 'mse':
        return StandardMSELoss()
    elif loss_name == 'custom_masked':
        return CustomLossWithMask(
            lambda_l2=cfg['training'].get('lambda_l2', 1.0),
            lambda_mask=cfg['training'].get('lambda_mask', 0.1)
        )
    elif loss_name == 'multi_mode_weighted':
        # 多模态加权 MSE
        lambdas = cfg['training'].get('lambdas', [])
        return MultiModeWeightedMSELoss(lambdas)
    else:
        raise ValueError(f"未知损失函数类型: {loss_name}")

def main():
    with open('modify_multi_attention/configs/config.yaml', 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, input_dim, output_dim = get_loaders(
        cfg["data"]["path"],
        cfg["data"]["batch_size"]
    )

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
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg["training"]["learning_rate"]
            )

            # 启动训练（不再传 use_mask/mask_boxes）
            trained_model, train_loss, valid_loss, test_loss = train_model(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=cfg["training"]["epochs"],
                device=device,
                attention_type=attn_type,
                config_path='modify_multi_attention/configs/config.yaml'
            )

            # 保存最优模型
            result_dir = os.path.join(parent_dir, attn_type)
            os.makedirs(result_dir, exist_ok=True)
            best_model_path = os.path.join(result_dir, f"best_model_{attn_type}.pt")
            torch.save(trained_model.state_dict(), best_model_path)

            # 可视化损失曲线
            if vis_enabled:
                plot_losses(train_loss, valid_loss, test_loss)
                loss_fig_path = os.path.join(result_dir, f"loss_curve_{attn_type}.png")
                plt.savefig(loss_fig_path)
                plt.close()

            # 测试并记录结果
            final_test_loss = test_model(
                trained_model,
                test_loader,
                criterion,
                device,
                attention_type=attn_type,
                parent_dir="attention_results",
                config_path='modify_multi_attention/configs/config.yaml'
            )

            with open(os.path.join(result_dir, f"test_result_{attn_type}.txt"), 'w') as fout:
                fout.write(f"Test Loss for {attn_type}: {final_test_loss}\n")

            print(f"✅ {attn_type} 训练完成！")

        except Exception as e:
            print(f"❌ 发生错误，跳过 {attn_type} 注意力机制")
            print(f"⚠️ 错误详情: {e}")
            failed_attention_types.append(attn_type)

    if failed_attention_types:
        with open(os.path.join(parent_dir, "failed_attention_log.txt"), "w") as fout:
            for attn in failed_attention_types:
                fout.write(f"{attn}\n")
        print("\n⚠️ 以下注意力机制训练失败，并已记录：")
        print("\n".join(failed_attention_types))
    else:
        print("\n🎉 所有注意力机制均运行成功！")

if __name__ == "__main__":
    main()  # 启动主函数
