from modify_multi_attention.utils.visualization import plot_difference_figure, plot_comparison_figure,plot_losses
import os
import torch
import matplotlib.pyplot as plt
import yaml


def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=100, device='cuda',
                early_stop_patience=10, attention_type='default', config_path='modify_multi_attention/configs/config.yaml',
                use_mask=False, mask_boxes=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model.to(device)

    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    best_valid_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for i, (in_press, out_pressure, time_steps, mask) in enumerate(train_loader):  # 解包为4个元素
            in_press, out_pressure, time_steps, mask = in_press.to(device), out_pressure.to(device), time_steps.to(device), mask.to(device)

            optimizer.zero_grad()
            model_out = model(in_press, time_steps)

            loss_value = criterion(model_out, out_pressure, mask)  # 传递掩码
            loss_value.backward()
            optimizer.step()
            total_train_loss += loss_value.item()

            # 可视化：每隔一定的epoch或批次绘制图像
            if cfg["visualization"]["enabled"] and (epoch + 1) % cfg["visualization"]["interval"] == 0 and i < cfg["visualization"]["max_samples"]:
                # 获取图像数据
                input_pressure = in_press[0].cpu().detach().numpy().reshape(20, 20)  # 假设数据形状为(20, 20)
                true_pressure = out_pressure[0].cpu().detach().numpy().reshape(200, 200)  # 假设数据形状为(200, 200)
                predicted_pressure = model_out[0].cpu().detach().numpy().reshape(200, 200)

                # 对比图
                plot_comparison_figure(
                    input_pressure=input_pressure,
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=time_steps[0].item(),
                    epoch=epoch,
                    idx=i,
                    attention_type=attention_type,
                    parent_dir="attention_results",
                    mode='train'
                )

                # 差异图
                plot_difference_figure(
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=time_steps[0].item(),
                    epoch=epoch,
                    idx=i,
                    attention_type=attention_type,
                    parent_dir="attention_results",
                    mode='train'
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # 可视化损失曲线
        if cfg["visualization"]["enabled"] and (epoch + 1) % cfg["visualization"]["interval"] == 0:
            result_dir = f"attention_results/{attention_type}"
            os.makedirs(result_dir, exist_ok=True)
            loss_fig_path = os.path.join(result_dir, f"loss_curve_epoch_{epoch + 1}.png")
            plot_losses(train_loss_history, valid_loss_history, test_loss_history, save_path=loss_fig_path)

        # 验证集评估
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for in_press, out_pressure, time_steps, mask in valid_loader:  # 解包为4个元素
                in_press, out_pressure, time_steps, mask = in_press.to(device), out_pressure.to(device), time_steps.to(device), mask.to(device)
                model_out = model(in_press, time_steps)
                loss_value = criterion(model_out, out_pressure, mask)  # 传递掩码
                total_valid_loss += loss_value.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_loss_history.append(avg_valid_loss)

        # 测试集评估
        total_test_loss = 0
        with torch.no_grad():
            for in_press, out_pressure, time_steps, mask in test_loader:  # 解包为4个元素
                in_press, out_pressure, time_steps, mask = in_press.to(device), out_pressure.to(device), time_steps.to(device), mask.to(device)
                model_out = model(in_press, time_steps)
                loss_value = criterion(model_out, out_pressure, mask)  # 传递掩码
                total_test_loss += loss_value.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(f"🎯 Epoch [{epoch + 1}/{num_epochs}] - Train: {avg_train_loss:.6f}, Valid: {avg_valid_loss:.6f}, Test: {avg_test_loss:.6f}")

    return model, train_loss_history, valid_loss_history, test_loss_history

def test_model(model, test_loader, criterion, device='cuda', attention_type='default', parent_dir="attention_results",
               config_path='modify_multi_attention/configs/config.yaml'):

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model.eval()
    model.to(device)
    total_test_loss = 0

    # 创建保存测试损失的日志文件
    loss_log_dir = f"attention_results/{attention_type}/loss_logs"
    os.makedirs(loss_log_dir, exist_ok=True)
    loss_log_path = os.path.join(loss_log_dir, "test_loss_log.txt")

    # 写入文件头
    with open(loss_log_path, 'w') as log_file:
        log_file.write("Batch, Test Loss\n")

    with torch.no_grad():
        for idx, (in_press, out_pressure, time_steps, mask) in enumerate(test_loader):  # 解包为4个元素
            in_press, out_pressure, time_steps, mask = (
                in_press.to(device), out_pressure.to(device), time_steps.to(device), mask.to(device)
            )

            model_out = model(in_press, time_steps)
            loss_value = criterion(model_out, out_pressure, mask)
            total_test_loss += loss_value.item()

            # 记录每个批次的测试损失到日志文件
            with open(loss_log_path, 'a') as log_file:
                log_file.write(f"{idx + 1}, {loss_value.item():.6f}\n")

            # 可视化对比图和差异图
            if cfg["visualization"]["enabled"] and idx < cfg["visualization"]["max_samples"]:
                input_pressure = in_press[0].view(20, 20).cpu().numpy()
                true_pressure = out_pressure[0].view(200, 200).cpu().numpy()
                predicted_pressure = model_out[0].view(200, 200).cpu().numpy()

                # 对比图
                plot_comparison_figure(
                    input_pressure=input_pressure,
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=time_steps[0].item(),
                    epoch=0,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=parent_dir,
                    mode='test'
                )

                # 差异图
                plot_difference_figure(
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=time_steps[0].item(),
                    epoch=0,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=parent_dir,
                    mode='test'
                )

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"🧪 测试完成，{attention_type} Test Loss: {avg_test_loss:.6f}")

    # 记录测试损失（平均损失）到日志文件
    with open(loss_log_path, 'a') as log_file:
        log_file.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    return avg_test_loss
