import os
import torch
import matplotlib.pyplot as plt
from modify_multi_attention.utils.visualization import plot_comparison_figure
from modify_multi_attention.utils.visualization import plot_difference_figure
from modify_multi_attention.utils.visualization import plot_losses

def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=100, device='cuda',
                early_stop_patience=10, attention_type='default',
                result_dir=None, cfg=None):   # cfg参数必传！

    # 标准 MSELoss（保证横向可比）
    import torch.nn as nn
    mse_loss = nn.MSELoss()

    # 配置参数直接来自cfg
    vis_enabled = cfg["visualization"]["enabled"]
    vis_interval = cfg["visualization"]["interval"]
    max_samples = cfg["visualization"]["max_samples"]

    model.to(device)

    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    best_valid_loss = float('inf')
    patience_counter = 0

    if result_dir is not None:
        loss_log_dir = os.path.join(result_dir, "loss_logs")
        os.makedirs(loss_log_dir, exist_ok=True)
        loss_log_path = os.path.join(loss_log_dir, "loss_log.txt")
        checkpoint_path = os.path.join(result_dir, f"checkpoint_{attention_type}.pth")
        save_dir = result_dir
    else:
        loss_log_dir = f"attention_results/{attention_type}/loss_logs"
        os.makedirs(loss_log_dir, exist_ok=True)
        loss_log_path = os.path.join(loss_log_dir, "loss_log.txt")
        checkpoint_path = f"attention_results/{attention_type}/checkpoint_{attention_type}.pth"
        save_dir = f"attention_results/{attention_type}"
        os.makedirs(save_dir, exist_ok=True)

    print(f"写入loss_log.txt到：{loss_log_path}")

    # ========== 恢复断点 ==========
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"检测到断点文件，自动恢复：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_loss_history = checkpoint.get('train_loss_history', [])
        valid_loss_history = checkpoint.get('valid_loss_history', [])
        test_loss_history  = checkpoint.get('test_loss_history', [])
        best_valid_loss    = checkpoint.get('best_valid_loss', float('inf'))
        patience_counter   = checkpoint.get('patience_counter', 0)
        start_epoch        = checkpoint.get('epoch', 0) + 1
        print(f"已恢复到 epoch {start_epoch}，best_valid_loss={best_valid_loss}")
    else:
        with open(loss_log_path, 'w') as log_file:
            log_file.write("Epoch, Train Loss, Valid Loss, Test Loss\n")

    # ========== 主训练循环 ==========
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0

        # ===== 训练用自定义 loss =====
        for i, (in_press, out_pressure, time_steps) in enumerate(train_loader):
            in_press, out_pressure, time_steps = (
                in_press.to(device), out_pressure.to(device), time_steps.to(device)
            )

            optimizer.zero_grad()
            model_out = model(in_press, time_steps)
            loss_value = criterion(model_out, out_pressure)  # 训练用自定义loss
            loss_value.backward()
            optimizer.step()
            total_train_loss += loss_value.item()

            if (i + 1) % 50 == 0 or i == 0:
                print(
                    f"    🔄 Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss_value.item():.6f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # ===== 验证用标准MSE =====
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for in_press, out_pressure, time_steps in valid_loader:
                in_press, out_pressure, time_steps = (
                    in_press.to(device), out_pressure.to(device), time_steps.to(device)
                )
                model_out = model(in_press, time_steps)
                loss_value = mse_loss(model_out, out_pressure)  # 验证横向对比只用MSE
                total_valid_loss += loss_value.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_loss_history.append(avg_valid_loss)

        # ===== 测试用标准MSE =====
        total_test_loss = 0
        with torch.no_grad():
            for in_press, out_pressure, time_steps in test_loader:
                in_press, out_pressure, time_steps = (
                    in_press.to(device), out_pressure.to(device), time_steps.to(device)
                )
                model_out = model(in_press, time_steps)
                loss_value = mse_loss(model_out, out_pressure)  # 测试也用MSE
                total_test_loss += loss_value.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(
            f"🎯 Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

        with open(loss_log_path, 'a') as log_file:
            log_file.write(f"{epoch + 1}, {avg_train_loss:.6f}, {avg_valid_loss:.6f}, {avg_test_loss:.6f}\n")

        # ========== 保存断点 ==========
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss_history,
            'valid_loss_history': valid_loss_history,
            'test_loss_history': test_loss_history,
            'best_valid_loss': best_valid_loss,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint, checkpoint_path)

        # ========== 保存最优模型 ==========
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_{attention_type}.pt"))
            print("✅ 模型已保存 (Best Model Updated)")
        else:
            patience_counter += 1
            print(f"⚠️ 早停计数: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            print("⏹️ 触发 Early Stopping!")
            break

        if vis_enabled and (epoch + 1) % vis_interval == 0:
            model.eval()
            with torch.no_grad():
                try:
                    sample_loader = iter(valid_loader)
                    for idx in range(min(max_samples, len(valid_loader))):
                        sample_input, sample_output, sample_time_steps = next(sample_loader)
                        sample_input, sample_output, sample_time_steps = (
                            sample_input.to(device),
                            sample_output.to(device),
                            sample_time_steps.to(device)
                        )
                        predictions = model(sample_input, sample_time_steps)

                        input_pressure = sample_input[0].view(20, 20).cpu().numpy()
                        true_pressure = sample_output[0].view(200, 200).cpu().numpy()
                        predicted_pressure = predictions[0].view(200, 200).cpu().numpy()

                        plot_comparison_figure(
                            input_pressure=input_pressure,
                            true_pressure=true_pressure,
                            predicted_pressure=predicted_pressure,
                            time_step=sample_time_steps[0].item(),
                            epoch=epoch + 1,
                            idx=idx,
                            attention_type=attention_type,
                            parent_dir=save_dir,
                            mode='validation'
                        )

                        plot_difference_figure(
                            true_pressure=true_pressure,
                            predicted_pressure=predicted_pressure,
                            time_step=sample_time_steps[0].item(),
                            epoch=epoch + 1,
                            idx=idx,
                            attention_type=attention_type,
                            parent_dir=save_dir,
                            mode='validation'
                        )

                except StopIteration:
                    print("⚠️ 验证集数据不足，无法生成可视化结果。")

        if (epoch + 1) % vis_interval == 0:
            plot_dir = os.path.join(save_dir, "loss_plots")
            os.makedirs(plot_dir, exist_ok=True)
            loss_fig_path = os.path.join(plot_dir, f"loss_curve_epoch_{epoch + 1}.png")
            plot_losses(train_loss_history, valid_loss_history, test_loss_history, save_path=loss_fig_path)

    plt.ioff()
    return model, train_loss_history, valid_loss_history, test_loss_history

def test_model(model, test_loader, criterion, device='cuda', attention_type='default', parent_dir=None, cfg=None):
    # 保证测试阶段只用MSELoss
    import torch.nn as nn
    mse_loss = nn.MSELoss()

    vis_enabled = cfg["visualization"]["enabled"]
    max_samples = cfg["visualization"]["max_samples"]

    model.eval()
    model.to(device)
    total_test_loss = 0

    if parent_dir is not None:
        loss_log_dir = os.path.join(parent_dir, "loss_logs")
    else:
        loss_log_dir = f"attention_results/{attention_type}/loss_logs"
    os.makedirs(loss_log_dir, exist_ok=True)
    loss_log_path = os.path.join(loss_log_dir, "test_loss_log.txt")

    with open(loss_log_path, 'w') as log_file:
        log_file.write("Batch, Test Loss\n")

    with torch.no_grad():
        for idx, (in_press, out_pressure, time_steps) in enumerate(test_loader):
            in_press, out_pressure, time_steps = (
                in_press.to(device), out_pressure.to(device), time_steps.to(device)
            )

            model_out = model(in_press, time_steps)
            loss_value = mse_loss(model_out, out_pressure)  # 只用MSE
            total_test_loss += loss_value.item()

            with open(loss_log_path, 'a') as log_file:
                log_file.write(f"{idx + 1}, {loss_value.item():.6f}\n")

            if vis_enabled and idx < max_samples:
                input_pressure = in_press[0].view(20, 20).cpu().numpy()
                true_pressure = out_pressure[0].view(200, 200).cpu().numpy()
                predicted_pressure = model_out[0].view(200, 200).cpu().numpy()

                plot_comparison_figure(
                    input_pressure=input_pressure,
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=time_steps[0].item(),
                    epoch=0,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=parent_dir if parent_dir is not None else f"attention_results/{attention_type}",
                    mode='test'
                )

                plot_difference_figure(
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=time_steps[0].item(),
                    epoch=0,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=parent_dir if parent_dir is not None else f"attention_results/{attention_type}",
                    mode='test'
                )

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"🧪 测试完成，{attention_type} Test Loss: {avg_test_loss:.6f}")

    with open(loss_log_path, 'a') as log_file:
        log_file.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    return avg_test_loss
