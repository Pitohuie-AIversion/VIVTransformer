import os
import yaml
import torch
import matplotlib.pyplot as plt
from modify_multi_attention.utils.visualization import (
    plot_difference_figure,
    plot_comparison_figure,
    plot_losses
)

def train_model(
    model,
    train_loader,
    valid_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs=100,
    device='cuda',
    attention_type='default',
    config_path='modify_multi_attention/configs/config.yaml'
):
    # 1. 读取配置，判断是否启用多模态加权
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    mode_enabled = cfg['training'].get('mask_weight_modes_enabled', False)

    model.to(device)
    train_loss_history, valid_loss_history, test_loss_history = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for i, batch in enumerate(train_loader):
            # 解包 batch（in_press, out_pressure, x_time_steps, mask, weight_maps）
            in_press, out_pressure, x_time_steps, mask, weight_maps = batch

            # 移动到设备
            in_press     = in_press.to(device)
            out_pressure = out_pressure.to(device)
            x_time_steps = x_time_steps.to(device)
            mask         = mask.to(device)
            weight_maps  = weight_maps.to(device)

            optimizer.zero_grad()
            # 前向：带时间步输入
            model_out = model(in_press, x_time_steps)

            # 根据模式启用不同 Loss
            if mode_enabled:
                loss_value = criterion(model_out, out_pressure, mask, weight_maps)
            else:
                loss_value = criterion(model_out, out_pressure, mask)

            loss_value.backward()
            optimizer.step()
            total_train_loss += loss_value.item()

            # 可视化（对比图 & 差异图）
            if cfg["visualization"]["enabled"] \
               and (epoch + 1) % cfg["visualization"]["interval"] == 0 \
               and i < cfg["visualization"]["max_samples"]:

                inp       = in_press[0].cpu().detach().numpy().reshape(20, 20)
                true      = out_pressure[0].cpu().detach().numpy().reshape(200, 200)
                pred      = model_out[0].cpu().detach().numpy().reshape(200, 200)
                tstep_val = x_time_steps[0].item()

                plot_comparison_figure(
                    inp, true, pred,
                    reynolds_number=None,
                    time_step=tstep_val,
                    epoch=epoch,
                    idx=i,
                    attention_type=attention_type,
                    mode='train'
                )
                plot_difference_figure(
                    true, pred,
                    reynolds_number=None,
                    time_step=tstep_val,
                    epoch=epoch,
                    idx=i,
                    attention_type=attention_type,
                    mode='train'
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # 可视化 Loss 曲线
        if cfg["visualization"]["enabled"] and (epoch + 1) % cfg["visualization"]["interval"] == 0:
            result_dir = f"attention_results/{attention_type}"
            os.makedirs(result_dir, exist_ok=True)
            loss_fig = os.path.join(result_dir, f"loss_curve_epoch_{epoch+1}.png")
            plot_losses(train_loss_history, valid_loss_history, test_loss_history, save_path=loss_fig)

        # 验证集评估
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                in_press, out_pressure, x_time_steps, mask, weight_maps = batch
                in_press     = in_press.to(device)
                out_pressure = out_pressure.to(device)
                x_time_steps = x_time_steps.to(device)
                mask         = mask.to(device)
                weight_maps  = weight_maps.to(device)

                model_out = model(in_press, x_time_steps)
                if mode_enabled:
                    loss_value = criterion(model_out, out_pressure, mask, weight_maps)
                else:
                    loss_value = criterion(model_out, out_pressure, mask)
                total_valid_loss += loss_value.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_loss_history.append(avg_valid_loss)

        # 测试集评估
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                in_press, out_pressure, x_time_steps, mask, weight_maps = batch
                in_press     = in_press.to(device)
                out_pressure = out_pressure.to(device)
                x_time_steps = x_time_steps.to(device)
                mask         = mask.to(device)
                weight_maps  = weight_maps.to(device)

                model_out = model(in_press, x_time_steps)
                if mode_enabled:
                    loss_value = criterion(model_out, out_pressure, mask, weight_maps)
                else:
                    loss_value = criterion(model_out, out_pressure, mask)
                total_test_loss += loss_value.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(
            f"🎯 Epoch [{epoch+1}/{num_epochs}] - "
            f"Train: {avg_train_loss:.6f}, "
            f"Valid: {avg_valid_loss:.6f}, "
            f"Test:  {avg_test_loss:.6f}"
        )

    return model, train_loss_history, valid_loss_history, test_loss_history


def test_model(
    model,
    test_loader,
    criterion,
    device='cuda',
    attention_type='default',
    parent_dir="attention_results",
    config_path='modify_multi_attention/configs/config.yaml'
):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    mode_enabled = cfg['training'].get('mask_weight_modes_enabled', False)

    model.eval()
    model.to(device)
    total_test_loss = 0.0

    # 日志准备
    log_dir = f"{parent_dir}/{attention_type}/loss_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "test_loss_log.txt")
    with open(log_path, 'w') as log_file:
        log_file.write("Batch, Test Loss\n")

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            in_press, out_pressure, x_time_steps, mask, weight_maps = batch
            in_press     = in_press.to(device)
            out_pressure = out_pressure.to(device)
            x_time_steps = x_time_steps.to(device)
            mask         = mask.to(device)
            weight_maps  = weight_maps.to(device)

            model_out = model(in_press, x_time_steps)
            if mode_enabled:
                loss_value = criterion(model_out, out_pressure, mask, weight_maps)
            else:
                loss_value = criterion(model_out, out_pressure, mask)
            total_test_loss += loss_value.item()

            # 记录日志
            with open(log_path, 'a') as log_file:
                log_file.write(f"{idx+1}, {loss_value.item():.6f}\n")

            # 可视化（同原逻辑）
            if cfg["visualization"]["enabled"] and idx < cfg["visualization"]["max_samples"]:
                inp       = in_press[0].view(20, 20).cpu().numpy()
                true      = out_pressure[0].view(200, 200).cpu().numpy()
                pred      = model_out[0].view(200, 200).cpu().numpy()
                tstep_val = x_time_steps[0].item()

                plot_comparison_figure(
                    input_pressure=inp,
                    true_pressure=true,
                    predicted_pressure=pred,
                    reynolds_number=None,
                    time_step=tstep_val,
                    epoch=0,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=parent_dir,
                    mode='test'
                )
                plot_difference_figure(
                    true_pressure=true,
                    predicted_pressure=pred,
                    reynolds_number=None,
                    time_step=tstep_val,
                    epoch=0,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=parent_dir,
                    mode='test'
                )

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"🧪 测试完成，{attention_type} Test Loss: {avg_test_loss:.6f}")

    with open(log_path, 'a') as log_file:
        log_file.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    return avg_test_loss
