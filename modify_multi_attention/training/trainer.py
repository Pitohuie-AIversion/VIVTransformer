from modify_multi_attention.utils.visualization import (
    plot_difference_figure, plot_comparison_figure, plot_losses,
    compare_svd_experiments, plot_svd_mode_grid_comparison,
    plot_top_n_svd_modes, plot_energy_and_mse_curves
)
import os
import torch
import matplotlib.pyplot as plt
import yaml
import numpy as np
import gc

def unpack_batch(batch):
    """
    保证兼容所有 collate_fn 返回的 batch（2、4、5元组等）
    """
    if batch is None:
        return (None,)*5
    if len(batch) == 5:
        return batch
    elif len(batch) == 4:
        return (*batch, None)
    elif len(batch) == 2:
        return (*batch, None, None, None)
    else:
        raise ValueError(f"Batch 不支持长度 {len(batch)}: {type(batch)}")

def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=100, device='cuda',
                early_stop_patience=10, attention_type='default',
                config_path='modify_multi_attention/configs/config.yaml',
                use_mask=False, mask_boxes=None):
    print(f"当前使用的loss类型: {criterion}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model.to(device)

    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []
    train_main_mode_loss_history = []
    valid_main_mode_loss_history = []
    test_main_mode_loss_history = []

    best_valid_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_main_mode_loss = 0
        n_train_batch = 0

        for i, batch in enumerate(train_loader):
            in_press, out_pressure, time_steps, mask, reynolds_idxs = unpack_batch(batch)
            if in_press is None or out_pressure is None:
                continue
            in_press, out_pressure, time_steps, mask = (
                in_press.to(device), out_pressure.to(device), time_steps.to(device), mask.to(device)
            )
            optimizer.zero_grad()
            model_out = model(in_press, time_steps)

            # 新loss支持多参数的自动判定
            try:
                if isinstance(criterion, tuple):
                    base_loss, svd_loss = criterion
                    loss_value = base_loss(model_out, out_pressure, mask)
                    loss_main = svd_loss(model_out, out_pressure)
                    total_loss = loss_value + loss_main
                    total_train_main_mode_loss += loss_main.item()
                else:
                    try:
                        # 先尝试(pred, target, mask)
                        total_loss = loss_value = criterion(model_out, out_pressure, mask)
                    except TypeError:
                        # 如果loss只接受两个参数（新SVD主模态loss等）
                        total_loss = loss_value = criterion(model_out, out_pressure)
                    loss_main = torch.tensor(0.0)
            except Exception as e:
                print(f"[Batch {i}] Exception in loss: {e}")
                continue

            total_train_loss += loss_value.item()
            n_train_batch += 1

            total_loss.backward()
            optimizer.step()

            # 可视化部分
            if cfg["visualization"]["enabled"] and (epoch + 1) % cfg["visualization"]["interval"] == 0 and i < cfg["visualization"]["max_samples"]:
                rp = reynolds_idxs[0].item() if reynolds_idxs is not None else 0
                ts = time_steps[0].item()
                input_pressure = in_press[0].cpu().detach().numpy().reshape(20, 20)
                true_pressure = out_pressure[0].cpu().detach().numpy().reshape(200, 200)
                pred_pressure = model_out[0].cpu().detach().numpy().reshape(200, 200)
                plot_comparison_figure(
                    input_pressure, true_pressure, pred_pressure, rp, ts, epoch, i, attention_type, "attention_results", "train"
                )
                plot_difference_figure(
                    true_pressure, pred_pressure, rp, ts, epoch, i, attention_type, "attention_results", "train"
                )
            # 内存释放
            del in_press, out_pressure, time_steps, mask, model_out
            torch.cuda.empty_cache()

        # 均值按实际 batch 数
        avg_train_loss = total_train_loss / (n_train_batch if n_train_batch else 1)
        train_loss_history.append(avg_train_loss)
        avg_train_main_mode_loss = total_train_main_mode_loss / (n_train_batch if n_train_batch else 1)
        train_main_mode_loss_history.append(avg_train_main_mode_loss)

        if cfg["visualization"]["enabled"] and (epoch + 1) % cfg["visualization"]["interval"] == 0:
            result_dir = f"attention_results/{attention_type}"
            os.makedirs(result_dir, exist_ok=True)
            loss_fig_path = os.path.join(result_dir, f"loss_curve_epoch_{epoch + 1}.png")
            if len(train_loss_history):
                plot_losses(train_loss_history, valid_loss_history, test_loss_history, save_path=loss_fig_path)
            if avg_train_main_mode_loss > 0 and len(train_main_mode_loss_history):
                plt.figure(figsize=(8, 4))
                plt.plot(train_main_mode_loss_history, label='Train Main Mode Loss')
                plt.xlabel('Epoch'); plt.ylabel('Main Mode Loss'); plt.legend()
                plt.title('Main Mode Loss Curve')
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, f"main_mode_loss_curve_{epoch + 1}.png"))
                plt.close()

        # 验证阶段
        model.eval()
        total_valid_loss = 0
        total_valid_main_mode_loss = 0
        n_valid_batch = 0
        with torch.no_grad():
            for batch in valid_loader:
                in_press, out_pressure, time_steps, mask, reynolds_idxs = unpack_batch(batch)
                if in_press is None or out_pressure is None:
                    continue
                in_press, out_pressure, time_steps, mask = (
                    in_press.to(device), out_pressure.to(device), time_steps.to(device), mask.to(device)
                )
                try:
                    if isinstance(criterion, tuple):
                        base_loss, svd_loss = criterion
                        loss_value = base_loss(model_out, out_pressure, mask)
                        loss_main = svd_loss(model_out, out_pressure)
                        total_valid_main_mode_loss += loss_main.item()
                    else:
                        try:
                            loss_value = criterion(model_out, out_pressure, mask)
                        except TypeError:
                            loss_value = criterion(model_out, out_pressure)
                        loss_main = torch.tensor(0.0)
                except Exception as e:
                    print(f"[Valid] Exception in loss: {e}")
                    continue

                total_valid_loss += loss_value.item()
                n_valid_batch += 1
                del in_press, out_pressure, time_steps, mask  # model_out
                torch.cuda.empty_cache()
        avg_valid_loss = total_valid_loss / (n_valid_batch if n_valid_batch else 1)
        valid_loss_history.append(avg_valid_loss)
        avg_valid_main_mode_loss = total_valid_main_mode_loss / (n_valid_batch if n_valid_batch else 1)
        valid_main_mode_loss_history.append(avg_valid_main_mode_loss)

        # 测试阶段
        total_test_loss = 0
        total_test_main_mode_loss = 0
        n_test_batch = 0
        with torch.no_grad():
            for batch in test_loader:
                in_press, out_pressure, time_steps, mask, reynolds_idxs = unpack_batch(batch)
                if in_press is None or out_pressure is None:
                    continue
                in_press, out_pressure, time_steps, mask = (
                    in_press.to(device), out_pressure.to(device), time_steps.to(device), mask.to(device)
                )
                try:
                    if isinstance(criterion, tuple):
                        base_loss, svd_loss = criterion
                        loss_value = base_loss(model_out, out_pressure, mask)
                        loss_main = svd_loss(model_out, out_pressure)
                        total_test_main_mode_loss += loss_main.item()
                    else:
                        try:
                            loss_value = criterion(model_out, out_pressure, mask)
                        except TypeError:
                            loss_value = criterion(model_out, out_pressure)
                        loss_main = torch.tensor(0.0)
                except Exception as e:
                    print(f"[Test] Exception in loss: {e}")
                    continue

                total_test_loss += loss_value.item()
                n_test_batch += 1
                del in_press, out_pressure, time_steps, mask
                torch.cuda.empty_cache()
        avg_test_loss = total_test_loss / (n_test_batch if n_test_batch else 1)
        test_loss_history.append(avg_test_loss)
        avg_test_main_mode_loss = total_test_main_mode_loss / (n_test_batch if n_test_batch else 1)
        test_main_mode_loss_history.append(avg_test_main_mode_loss)

        print(f"🎯 Epoch [{epoch + 1}/{num_epochs}] - Train: {avg_train_loss:.6f}, Valid: {avg_valid_loss:.6f}, Test: {avg_test_loss:.6f} | Main Mode Train: {avg_train_main_mode_loss:.6f}")
        gc.collect()
        torch.cuda.empty_cache()

    # 保存loss曲线和主模态loss曲线
    result_dir = f"attention_results/{attention_type}"
    np.savetxt(os.path.join(result_dir, "train_loss_curve.txt"), np.array(train_loss_history))
    np.savetxt(os.path.join(result_dir, "valid_loss_curve.txt"), np.array(valid_loss_history))
    np.savetxt(os.path.join(result_dir, "test_loss_curve.txt"), np.array(test_loss_history))
    np.savetxt(os.path.join(result_dir, "train_main_mode_loss_curve.txt"), np.array(train_main_mode_loss_history))
    np.savetxt(os.path.join(result_dir, "valid_main_mode_loss_curve.txt"), np.array(valid_main_mode_loss_history))
    np.savetxt(os.path.join(result_dir, "test_main_mode_loss_curve.txt"), np.array(test_main_mode_loss_history))

    return model, train_loss_history, valid_loss_history, test_loss_history

def test_model(model, test_loader, criterion, device='cuda', attention_type='default', parent_dir="attention_results",
               config_path='modify_multi_attention/configs/config.yaml'):
    from modify_multi_attention.utils.visualization import (
        compare_svd_experiments, plot_svd_mode_grid_comparison,
        plot_top_n_svd_modes, plot_energy_and_mse_curves
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model.eval()
    model.to(device)
    total_test_loss = 0
    n_test_batch = 0

    loss_log_dir = f"attention_results/{attention_type}/loss_logs"
    os.makedirs(loss_log_dir, exist_ok=True)
    loss_log_path = os.path.join(loss_log_dir, "test_loss_log.txt")

    do_svd_vis = cfg["visualization"].get("do_svd_visualization", True)
    svd_vis_num = cfg["visualization"].get("svd_vis_num", 5)
    h, w = cfg["model"].get("output_h", 200), cfg["model"].get("output_w", 200)
    pred_list, target_list = [], []

    with open(loss_log_path, 'w') as log_file:
        log_file.write("Batch, Test Loss\n")

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            in_press, out_pressure, time_steps, mask, reynolds_idxs = unpack_batch(batch)
            if in_press is None or out_pressure is None:
                continue
            in_press, out_pressure, time_steps, mask = (
                in_press.to(device), out_pressure.to(device), time_steps.to(device), mask.to(device)
            )
            model_out = model(in_press, time_steps)  # 🔔 必须加上这句
            try:
                try:
                    loss_value = criterion(model_out, out_pressure, mask)
                except TypeError:
                    loss_value = criterion(model_out, out_pressure)
            except Exception as e:
                print(f"[TestModel] Exception in loss: {e}")
                continue

            total_test_loss += loss_value.item()
            n_test_batch += 1

            with open(loss_log_path, 'a') as log_file:
                log_file.write(f"{idx + 1}, {loss_value.item():.6f}\n")

            # 记录预测和真实，便于SVD批量分析
            if do_svd_vis and idx < svd_vis_num:
                pred_field = model_out[0].detach().cpu().numpy().reshape(h, w)
                target_field = out_pressure[0].detach().cpu().numpy().reshape(h, w)
                pred_list.append(pred_field)
                target_list.append(target_field)

            # 可视化
            if cfg["visualization"]["enabled"] and idx < cfg["visualization"]["max_samples"]:
                input_pressure = in_press[0].view(20, 20).cpu().numpy()
                true_pressure = out_pressure[0].view(200, 200).cpu().numpy()
                predicted_pressure = model_out[0].view(200, 200).cpu().numpy()
                rp = reynolds_idxs[0].item() if reynolds_idxs is not None else 0
                ts = time_steps[0].item() if time_steps.dim() > 0 else 0
                plot_comparison_figure(
                    input_pressure, true_pressure, predicted_pressure, rp, ts, 0, idx, attention_type, parent_dir, "test"
                )
                plot_difference_figure(
                    true_pressure, predicted_pressure, rp, ts, 0, idx, attention_type, parent_dir, "test"
                )
            del in_press, out_pressure, time_steps, mask, model_out
            torch.cuda.empty_cache()

    avg_test_loss = total_test_loss / (n_test_batch if n_test_batch else 1)
    print(f"🧪 测试完成，{attention_type} Test Loss: {avg_test_loss:.6f}")

    with open(loss_log_path, 'a') as log_file:
        log_file.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    # SVD分解可视化（不变）
    if do_svd_vis and len(pred_list) > 0:
        svd_vis_dir = os.path.join(parent_dir, attention_type, "svd_visualization")
        os.makedirs(svd_vis_dir, exist_ok=True)
        for i in range(len(pred_list)):
            compare_svd_experiments(
                pred_nosvd=pred_list[i], pred_withsvd=pred_list[i], target=target_list[i],
                h=h, w=w, save_path=os.path.join(svd_vis_dir, f"svd_spectrum_{i}.png")
            )
            plot_svd_mode_grid_comparison(
                pred_nosvd=pred_list[i], pred_withsvd=pred_list[i], target=target_list[i],
                h=h, w=w, n_modes=3,
                save_path=os.path.join(svd_vis_dir, f"svd_mode_grid_{i}.png")
            )
            plot_top_n_svd_modes(
                pred_list[i], n_modes=10,
                save_path=os.path.join(svd_vis_dir, f"svd_pred_top10_{i}.png")
            )
            plot_top_n_svd_modes(
                target_list[i], n_modes=10,
                save_path=os.path.join(svd_vis_dir, f"svd_target_top10_{i}.png")
            )
            plot_energy_and_mse_curves(
                pred_list[i], max_modes=10,
                save_path=os.path.join(svd_vis_dir, f"svd_pred_energy_{i}.png")
            )
            plot_energy_and_mse_curves(
                target_list[i], max_modes=10,
                save_path=os.path.join(svd_vis_dir, f"svd_target_energy_{i}.png")
            )

    gc.collect()
    torch.cuda.empty_cache()
    return avg_test_loss
