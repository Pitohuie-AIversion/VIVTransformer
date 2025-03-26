import torch
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib
from modify_multi_attention.data.dataloader import get_loaders
from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
from modify_multi_attention.training.trainer import train_model, test_model
from modify_multi_attention.utils.visualization import plot_losses

matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    # === 动态路径 ===
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "configs", "config.yaml")
    parent_dir = os.path.join(current_dir, "attention_results")
    os.makedirs(parent_dir, exist_ok=True)
    failed_log_path = os.path.join(parent_dir, "failed_attention_log.txt")

    # === 自动写入你指定的失败 attention ===
    predefined_failed_list =  [
    "psa", "sge", "aft", "outlook",
    "vip", "coatnet", "halo", "residual",
    "crossformer", "moa", "dat", "mobilevit", "mobilevitv2"
]

    with open(failed_log_path, "w") as f:
        for attn in predefined_failed_list:
            f.write(f"{attn}\n")
    print(f"✅ 已自动重写 failed_attention_log.txt，共 {len(predefined_failed_list)} 个 attention")

    # === 读取 config.yaml ===
    try:
        with open(yaml_path, 'r', encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {yaml_path}")
        return

    device = torch.device(cfg["device"])
    print(f"🖥️ Using device: {device}")

    # === 读取 failed log ===
    ATTENTION_TYPES = []
    if os.path.exists(failed_log_path):
        print(f"\n📄 读取失败记录: {failed_log_path}")
        with open(failed_log_path, "r") as f:
            for line in f:
                tokens = line.strip().split()  # 空格或换行
                ATTENTION_TYPES.extend(tokens)
        print(f"🔁 本次将重新测试以下 attention: {ATTENTION_TYPES}")
    else:
        print("\n⚠️ 没有检测到 failed_attention_log.txt，退出")
        return

    vis_enabled = cfg["visualization"]["enabled"]
    failed_attention_types = []

    # === 加载数据 ===
    train_loader, valid_loader, test_loader = get_loaders(cfg["data"]["path"], cfg["data"]["batch_size"])

    # === 遍历 attention 测试 ===
    for attn_type in ATTENTION_TYPES:
        print(f"\n=========== 测试注意力机制: {attn_type} ===========")

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

            # === 训练 ===
            trained_model, train_loss, valid_loss, test_loss = train_model(
                model, train_loader, valid_loader, test_loader,
                criterion, optimizer, cfg["training"]["epochs"],
                device, cfg["training"]["early_stop_patience"],
                attention_type=attn_type
            )

            # === 保存模型 ===
            result_dir = os.path.join(parent_dir, attn_type)
            os.makedirs(result_dir, exist_ok=True)
            torch.save(trained_model.state_dict(), os.path.join(result_dir, f"best_model_{attn_type}.pt"))

            # === 损失曲线 ===
            if vis_enabled:
                plot_losses(train_loss, valid_loss, test_loss)
                plt.savefig(os.path.join(result_dir, f"loss_curve_{attn_type}.png"))
                plt.close()

            # === 测试 ===
            final_test_loss = test_model(
                trained_model, test_loader, criterion, device,
                attention_type=attn_type, parent_dir=parent_dir
            )

            # === 保存 test 结果 ===
            with open(os.path.join(result_dir, f"test_result_{attn_type}.txt"), 'w') as f:
                f.write(f"Test Loss for {attn_type}: {final_test_loss}\n")

            print(f"✅ {attn_type} 测试完成！")

        except Exception as e:
            print(f"❌ {attn_type} 出现错误，跳过")
            print(f"⚠️ 错误详情: {str(e)}")
            failed_attention_types.append(attn_type)

    # === 更新失败记录 ===
    if failed_attention_types:
        with open(failed_log_path, "w") as f:
            for attn in failed_attention_types:
                f.write(f"{attn}\n")
        print(f"\n⚠️ 以下 Attention 机制仍失败，log 已更新：")
        print("\n".join(failed_attention_types))
    else:
        print("\n🎉 本次所有 Attention 均已成功")
        if os.path.exists(failed_log_path):
            os.remove(failed_log_path)
            print("✅ 已清除 failed_attention_log.txt")

if __name__ == "__main__":
    main()
