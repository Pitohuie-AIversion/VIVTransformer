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

import os
import yaml
import torch
import matplotlib.pyplot as plt
from modify_multi_attention.data.dataloader import get_loaders
from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
from modify_multi_attention.training.trainer import train_model, test_model
from modify_multi_attention.utils.visualization import plot_losses


def main():
    # 读取配置文件
    with open('modify_multi_attention/configs/config.yaml', 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"])
    print(f"Using device: {device}")

    # 读取 YAML 配置
    ATTENTION_TYPES = cfg["attention_types"]
    vis_enabled = cfg["visualization"]["enabled"]
    parent_dir = "attention_results"
    os.makedirs(parent_dir, exist_ok=True)

    failed_attention_types = []  # 记录失败的注意力机制

    train_loader, valid_loader, test_loader = get_loaders(cfg["data"]["path"], cfg["data"]["batch_size"])

    for attn_type in ATTENTION_TYPES:
        print(f"\n=========== 当前测试注意力机制: {attn_type} ===========")

        try:
            model = TransformerFlowReconstructionModel(
                input_dim=cfg["model"]["input_dim"],
                output_dim=cfg["model"]["output_dim"],
                num_heads=cfg["model"]["num_heads"],
                num_layers=cfg["model"]["num_layers"],
                d_model=cfg["model"]["d_model"],
                max_time_steps=cfg["model"]["max_time_steps"],
                attention_type=attn_type
            ).to(device)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

            # 训练模型
            trained_model, train_loss, valid_loss, test_loss = train_model(
                model, train_loader, valid_loader, test_loader,
                criterion, optimizer, cfg["training"]["epochs"],
                device, cfg["training"]["early_stop_patience"],
                attention_type=attn_type
            )

            # 创建存储路径
            result_dir = os.path.join(parent_dir, attn_type)
            os.makedirs(result_dir, exist_ok=True)

            # ✅ **保存训练好的最佳模型**
            best_model_path = os.path.join(result_dir, f"best_model_{attn_type}.pt")
            torch.save(trained_model.state_dict(), best_model_path)

            # ✅ **绘制损失曲线**
            if vis_enabled:
                plot_losses(train_loss, valid_loss, test_loss)
                loss_fig_path = os.path.join(result_dir, f"loss_curve_{attn_type}.png")
                plt.savefig(loss_fig_path)
                plt.close()

            # ✅ **测试模型**
            final_test_loss = test_model(
                trained_model, test_loader, criterion, device,
                attention_type=attn_type,
                parent_dir="attention_results"
            )

            # ✅ **保存测试结果**
            test_result_file = os.path.join(result_dir, f"test_result_{attn_type}.txt")
            with open(test_result_file, 'w') as f:
                f.write(f"Test Loss for {attn_type}: {final_test_loss}\n")

            print(f"✅ {attn_type} 训练完成！")

        except Exception as e:
            print(f"❌ 发生错误，跳过 {attn_type} 注意力机制")
            print(f"⚠️ 错误详情: {str(e)}")
            failed_attention_types.append(attn_type)

    # 记录失败的注意力机制
    if failed_attention_types:
        with open(os.path.join(parent_dir, "failed_attention_log.txt"), "w") as f:
            for attn in failed_attention_types:
                f.write(f"{attn}\n")

        print(f"\n⚠️ 以下注意力机制训练失败，并已记录在 failed_attention_log.txt：")
        print("\n".join(failed_attention_types))
    else:
        print("\n🎉 所有注意力机制均运行成功！")


if __name__ == "__main__":
    main()

