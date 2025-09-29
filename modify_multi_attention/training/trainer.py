import os
# 设置无头模式环境变量（必须在任何GUI相关导入之前）
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('DISPLAY', '')
os.environ.setdefault('HEADLESS', '1')

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
import json
from datetime import datetime
import math
import torch.nn.functional as F
from ..utils.visualization import plot_comparison_figure
from ..utils.visualization import plot_difference_figure
from ..utils.visualization import plot_losses

# 全局调试开关 - 可通过配置文件控制
DEBUG_MODE = True  # 默认开启，可通过配置文件关闭

# 设置matplotlib支持中文显示
# 统一使用全局 sitecustomize.py 的中文字体和负号设置
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
# from utils.visualization import plot_comparison_figure
# from utils.visualization import plot_difference_figure
# from utils.visualization import plot_losses

# ========== Finite 检查与日志工具函数（仅在发现非有限时写入） ==========
def _finite_stats(x):
    try:
        if isinstance(x, torch.Tensor):
            x_det = x.detach()
            isfinite = torch.isfinite(x_det)
            numel = x_det.numel()
            num_nan = torch.isnan(x_det).sum().item()
            num_pos_inf = torch.isposinf(x_det).sum().item() if hasattr(torch, 'isposinf') else ((x_det == float('inf')).sum().item())
            num_neg_inf = torch.isneginf(x_det).sum().item() if hasattr(torch, 'isneginf') else ((x_det == float('-inf')).sum().item())
            finite_vals = x_det[isfinite]
            stats = {
                'total': int(numel),
                'num_nan': int(num_nan),
                'num_pos_inf': int(num_pos_inf),
                'num_neg_inf': int(num_neg_inf),
                'num_not_finite': int(num_nan + num_pos_inf + num_neg_inf),
                'dtype': str(x_det.dtype),
                'shape': tuple(x_det.shape),
            }
            if finite_vals.numel() > 0:
                stats.update({
                    'min': float(finite_vals.min().item()),
                    'max': float(finite_vals.max().item()),
                    'mean': float(finite_vals.mean().item()),
                })
            else:
                stats.update({'min': None, 'max': None, 'mean': None})
            # 采样少量值用于排查
            try:
                sample = x_det.reshape(-1)[:10].tolist()
                stats['sample_first_10'] = [float(v) if isinstance(v, (int, float)) else float(v) for v in sample]
            except Exception:
                stats['sample_first_10'] = []
            return stats
        else:
            # 处理标量/数值
            val = float(x)
            return {
                'total': 1,
                'num_nan': 0 if math.isfinite(val) and not math.isnan(val) else (1 if math.isnan(val) else 0),
                'num_pos_inf': 1 if val == float('inf') else 0,
                'num_neg_inf': 1 if val == float('-inf') else 0,
                'num_not_finite': 0 if math.isfinite(val) else 1,
                'dtype': type(x).__name__,
                'shape': (),
                'min': val,
                'max': val,
                'mean': val,
                'sample_first_10': [val],
            }
    except Exception as e:
        return {'error': f'stats_failed: {e}'}


def _append_finite_record(log_path, stage, tag, tensor_or_scalar, epoch=None, batch_idx=None):
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        record = {
            'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stage': stage,
            'epoch': int(epoch) if epoch is not None else None,
            'batch': int(batch_idx) if batch_idx is not None else None,
            'tag': tag,
            'stats': _finite_stats(tensor_or_scalar),
        }
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"[finite_debug] Failed to write log: {e}")


def _log_tensor_if_not_finite(x, log_path, stage, tag, epoch=None, batch_idx=None):
    """仅在调试模式下检查tensor有限性"""
    if not DEBUG_MODE:
        return False
    
    try:
        if isinstance(x, torch.Tensor):
            if not torch.isfinite(x.detach()).all():
                _append_finite_record(log_path, stage, tag, x, epoch=epoch, batch_idx=batch_idx)
                return True
        else:
            # 标量/数值
            val = float(x)
            if not math.isfinite(val):
                _append_finite_record(log_path, stage, tag, val, epoch=epoch, batch_idx=batch_idx)
                return True
    except Exception as e:
        print(f"[finite_debug] check failed for {tag}: {e}")
    return False


def _inspect_forward_cnn_resize(model, inputs, finite_log_path, batch_idx: int = None):
    """
    针对 CNNResizeBaselineModel 的逐步前向检查：
    依次检查 x 组装、backbone、interpolate、head、view 各阶段的有限性。
    仅在调试模式下进行前向传播检查
    """
    if not DEBUG_MODE:
        return
    
    try:
        in_press, time_steps = inputs
        B = in_press.shape[0]
        # x reshape
        x = in_press.view(B, 1, model.H_in, model.W_in)
        _log_tensor_if_not_finite(x, finite_log_path, 'inspect', 'x_reshaped', batch_idx=batch_idx)
        # time concat
        if getattr(model, 'use_time', False) and time_steps is not None:
            tnorm = (time_steps.float() / max(1, int(getattr(model, 'max_time_steps', 100)) - 1)).view(B, 1, 1, 1)
            tmap = tnorm.expand(B, 1, model.H_in, model.W_in)
            x = torch.cat([x, tmap], dim=1)
        _log_tensor_if_not_finite(x, finite_log_path, 'inspect', 'x_after_time_concat', batch_idx=batch_idx)
        # backbone
        feat = model.backbone(x)
        _log_tensor_if_not_finite(feat, finite_log_path, 'inspect', 'feat_backbone', batch_idx=batch_idx)
        # interpolate
        feat_up = F.interpolate(feat, size=(model.H_out, model.W_out), mode="bilinear", align_corners=False)
        _log_tensor_if_not_finite(feat_up, finite_log_path, 'inspect', 'feat_after_interpolate', batch_idx=batch_idx)
        # head conv
        out_map = model.head(feat_up)
        _log_tensor_if_not_finite(out_map, finite_log_path, 'inspect', 'out_after_head', batch_idx=batch_idx)
        # view
        out_flat = out_map.view(B, -1)
        _log_tensor_if_not_finite(out_flat, finite_log_path, 'inspect', 'out_after_view', batch_idx=batch_idx)
    except Exception as e:
        if DEBUG_MODE:
            print(f"[finite_debug] CNNResize inspect failed: {e}")


def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, num_epochs=100, device='cuda',
                early_stop_patience=10, attention_type='default',
                result_dir=None, cfg=None, scheduler=None, no_pretrained=False):   # cfg参数必传！

    # 从配置文件读取调试模式设置
    global DEBUG_MODE
    experiment_config = cfg.get('experiment', {})
    DEBUG_MODE = experiment_config.get('debug_mode', True)
    
    if not DEBUG_MODE:
        print("调试模式已关闭，将跳过tensor有限性检查以提高训练速度")
    
    # 标准 MSELoss（保证横向可比）
    import torch.nn as nn
    mse_loss = nn.MSELoss()
    
    # 混合精度训练配置
    mixed_precision_config = cfg.get('mixed_precision', {})
    device_str = str(device) if hasattr(device, 'type') else str(device)
    use_amp = mixed_precision_config.get('enabled', False) and 'cuda' in device_str
    scaler = GradScaler('cuda') if use_amp else None
    
    if use_amp:
        print(f"Enabled AMP (mixed precision), loss scale: {mixed_precision_config.get('loss_scale', 'dynamic')}")
    
    # 梯度累积配置
    gradient_config = cfg.get('gradient', {})
    accumulation_steps = gradient_config.get('accumulation_steps', 1)
    if accumulation_steps > 1:
        print(f"Gradient accumulation enabled, steps: {accumulation_steps}")

    # 配置参数直接来自cfg
    vis_enabled = cfg["visualization"]["enabled"]
    vis_interval = cfg["visualization"]["interval"]
    max_samples = cfg["visualization"]["max_samples"]
    
    # 早停配置参数
    training_config = cfg.get("training", {})
    early_stopping_config = training_config.get("early_stopping", {})
    enable_early_stopping = early_stopping_config.get("enabled", False)  # 默认禁用早停
    patience = early_stopping_config.get("patience", early_stop_patience)
    min_delta = early_stopping_config.get("min_delta", 1e-6)
    monitor = early_stopping_config.get("monitor", "val_loss")
    mode = early_stopping_config.get("mode", "min")
    restore_best_weights = early_stopping_config.get("restore_best_weights", True)
    
    print(f"Early stopping config: enabled={enable_early_stopping}, monitor={monitor}, mode={mode}, patience={patience}")

    # 只有在模型不是DataParallel时才移动到device
    # DataParallel模型已经在主程序中正确设置了设备
    if not isinstance(model, torch.nn.DataParallel):
        model.to(device)

    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    # 早停相关变量
    if mode == "min":
        best_metric = float('inf')
        is_better = lambda current, best: current < (best - min_delta)
    else:  # mode == "max"
        best_metric = float('-inf')
        is_better = lambda current, best: current > (best + min_delta)
    
    patience_counter = 0
    best_model_state = None  # 用于保存最佳模型权重
    
    # 兼容旧版本
    best_valid_loss = float('inf')

    if result_dir is not None:
        loss_log_dir = os.path.join(result_dir, "loss_logs")
        os.makedirs(loss_log_dir, exist_ok=True)
        loss_log_path = os.path.join(loss_log_dir, "loss_log.txt")
        checkpoint_path = os.path.join(result_dir, f"checkpoint_{attention_type}.pth")
        save_dir = result_dir
        # 确保保存目录存在（用于写入 finite_debug.txt 等文件）
        os.makedirs(save_dir, exist_ok=True)
    else:
        loss_log_dir = f"attention_results/{attention_type}/loss_logs"
        os.makedirs(loss_log_dir, exist_ok=True)
        loss_log_path = os.path.join(loss_log_dir, "loss_log.txt")
        checkpoint_path = f"attention_results/{attention_type}/checkpoint_{attention_type}.pth"
        save_dir = f"attention_results/{attention_type}"
        os.makedirs(save_dir, exist_ok=True)

    # finite 调试日志路径
    finite_log_path = os.path.join(save_dir, "finite_debug.txt")
    # 标记一次新的run（确保文件被创建）
    try:
        with open(finite_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'event': 'run_start', 'attention_type': attention_type}) + '\n')
    except Exception as e:
        print(f"[finite_debug] failed to init log {finite_log_path}: {e}")

    print(f"Writing loss_log.txt to: {loss_log_path}")

    # ========== 恢复断点 ==========
    start_epoch = 0
    if not no_pretrained and os.path.exists(checkpoint_path):
        print(f"Found checkpoint; auto-resume: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_loss_history = checkpoint.get('train_loss_history', [])
            valid_loss_history = checkpoint.get('valid_loss_history', [])
            test_loss_history  = checkpoint.get('test_loss_history', [])
            best_valid_loss    = checkpoint.get('best_valid_loss', float('inf'))
            best_metric        = checkpoint.get('best_metric', best_metric)
            patience_counter   = checkpoint.get('patience_counter', 0)
            best_model_state   = checkpoint.get('best_model_state', None)
            start_epoch        = checkpoint.get('epoch', 0) + 1
            print(f"Resumed to epoch {start_epoch}, best_metric={best_metric}, patience_counter={patience_counter}")
            # 加载后做一次参数有限性体检
            bad_params = []
            for name, p in model.named_parameters():
                if p is not None and (not torch.isfinite(p.detach()).all()):
                    bad_params.append(name)
            if bad_params:
                print(f"[finite_debug][WARN] Non-finite params detected after resume: {bad_params}")
                try:
                    os.remove(checkpoint_path)
                    print(f"[finite_debug] Removed corrupted checkpoint: {checkpoint_path}")
                except Exception as e_rm:
                    print(f"[finite_debug] Failed to remove checkpoint: {e_rm}")
                # 回退到从头训练
                start_epoch = 0
                best_metric = float('inf') if mode == 'min' else float('-inf')
                patience_counter = 0
                best_model_state = None
                train_loss_history, valid_loss_history, test_loss_history = [], [], []
                print("Will ignore checkpoint and train from scratch...")
        except RuntimeError as e:
            print(f"Warning: failed to load pretrained model: {e}")
            print("Will train from scratch...")
    elif no_pretrained:
        print("Skip pretrained model loading; train from scratch")
    else:
        print("No checkpoint found; train from scratch")
    
    # 初始化损失日志文件
    with open(loss_log_path, 'w') as log_file:
        log_file.write("Epoch, Train Loss, Valid Loss, Test Loss\n")

    # ========== 主训练循环 ==========
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0

        # ===== 训练用自定义 loss =====
        for i, (in_press, out_pressure, time_steps) in enumerate(train_loader):
            # CPU数据加载优化：只在需要时传输到GPU，减少GPU内存占用
            if device.type == 'cuda':
                # 非阻塞传输，提高效率
                in_press = in_press.to(device, non_blocking=True)
                out_pressure = out_pressure.to(device, non_blocking=True) 
                time_steps = time_steps.to(device, non_blocking=True)
            else:
                in_press, out_pressure, time_steps = (
                    in_press.to(device), out_pressure.to(device), time_steps.to(device)
                )

            # 输入/标签/时间步 finite 检查
            _log_tensor_if_not_finite(in_press, finite_log_path, 'train', 'input/in_press', epoch=epoch+1, batch_idx=i+1)
            _log_tensor_if_not_finite(out_pressure, finite_log_path, 'train', 'target/out_pressure', epoch=epoch+1, batch_idx=i+1)
            _log_tensor_if_not_finite(time_steps, finite_log_path, 'train', 'time_steps', epoch=epoch+1, batch_idx=i+1)

            # 梯度累积：只在累积步数的开始清零梯度
            if i % accumulation_steps == 0:
                optimizer.zero_grad()
            
            # 混合精度训练
            if use_amp:
                with autocast('cuda'):
                    model_out = model(in_press, time_steps)
                    _log_tensor_if_not_finite(model_out, finite_log_path, 'train', 'model_out', epoch=epoch+1, batch_idx=i+1)
                    loss_value = criterion(model_out, out_pressure)  # 训练用自定义loss
                    _log_tensor_if_not_finite(loss_value, finite_log_path, 'train', 'loss_value', epoch=epoch+1, batch_idx=i+1)
                    # 梯度累积：损失需要除以累积步数
                    loss_value = loss_value / accumulation_steps
                
                scaler.scale(loss_value).backward()
                
                # 梯度累积：只在累积步数结束时更新参数
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # 梯度裁剪
                    if gradient_config.get('clip_enabled', False):
                        clip_value = gradient_config.get('clip_value', 1.0)
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    
                    # 仅在存在有效梯度时才调用 step，避免 GradScaler 警告
                    has_grad = False
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                has_grad = True
                                break
                        if has_grad:
                            break
                    if has_grad:
                        scaler.step(optimizer)
                    else:
                        print("AMP: no valid gradients detected in this step; skipping optimizer.step() to avoid GradScaler warning")
                    scaler.update()
            else:
                model_out = model(in_press, time_steps)
                _log_tensor_if_not_finite(model_out, finite_log_path, 'train', 'model_out', epoch=epoch+1, batch_idx=i+1)
                loss_value = criterion(model_out, out_pressure)  # 训练用自定义loss
                _log_tensor_if_not_finite(loss_value, finite_log_path, 'train', 'loss_value', epoch=epoch+1, batch_idx=i+1)
                # 梯度累积：损失需要除以累积步数
                loss_value = loss_value / accumulation_steps
                loss_value.backward()
                
                # 梯度累积：只在累积步数结束时更新参数
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # 梯度裁剪
                    if gradient_config.get('clip_enabled', False):
                        clip_value = gradient_config.get('clip_value', 1.0)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    
                    optimizer.step()
            
            total_train_loss += loss_value.item() * accumulation_steps  # 恢复原始损失值用于记录

            if (i + 1) % 50 == 0 or i == 0:
                print(
                    f"    Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss_value.item() * accumulation_steps:.6f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # ===== 验证用标准MSE =====
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for j, (in_press, out_pressure, time_steps) in enumerate(valid_loader):
                # CPU数据加载优化：验证时也使用非阻塞传输
                if device.type == 'cuda':
                    in_press = in_press.to(device, non_blocking=True)
                    out_pressure = out_pressure.to(device, non_blocking=True)
                    time_steps = time_steps.to(device, non_blocking=True)
                else:
                    in_press, out_pressure, time_steps = (
                        in_press.to(device), out_pressure.to(device), time_steps.to(device)
                    )
                # 输入 finite 检查
                _log_tensor_if_not_finite(in_press, finite_log_path, 'valid', 'input/in_press', epoch=epoch+1, batch_idx=j+1)
                _log_tensor_if_not_finite(out_pressure, finite_log_path, 'valid', 'target/out_pressure', epoch=epoch+1, batch_idx=j+1)
                _log_tensor_if_not_finite(time_steps, finite_log_path, 'valid', 'time_steps', epoch=epoch+1, batch_idx=j+1)

                model_out = model(in_press, time_steps)
                _log_tensor_if_not_finite(model_out, finite_log_path, 'valid', 'model_out', epoch=epoch+1, batch_idx=j+1)
                loss_value = _safe_mse(model_out, out_pressure)  # 验证横向对比使用安全MSE
                _log_tensor_if_not_finite(loss_value, finite_log_path, 'valid', 'loss_value', epoch=epoch+1, batch_idx=j+1)
                total_valid_loss += loss_value.item()

        # 处理验证集为空的情况
        if len(valid_loader) > 0:
            avg_valid_loss = total_valid_loss / len(valid_loader)
            valid_loss_history.append(avg_valid_loss)
        else:
            avg_valid_loss = float('inf')  # 验证集为空时设置为无穷大
            valid_loss_history.append(avg_valid_loss)
            print("    Validation set is empty; skipping validation step")

        # ===== 测试用标准MSE =====
        total_test_loss = 0
        with torch.no_grad():
            for k, (in_press, out_pressure, time_steps) in enumerate(test_loader):
                # CPU数据加载优化：测试时也使用非阻塞传输
                if device.type == 'cuda':
                    in_press = in_press.to(device, non_blocking=True)
                    out_pressure = out_pressure.to(device, non_blocking=True)
                    time_steps = time_steps.to(device, non_blocking=True)
                else:
                    in_press, out_pressure, time_steps = (
                        in_press.to(device), out_pressure.to(device), time_steps.to(device)
                    )
                _log_tensor_if_not_finite(in_press, finite_log_path, 'test', 'input/in_press', epoch=epoch+1, batch_idx=k+1)
                _log_tensor_if_not_finite(out_pressure, finite_log_path, 'test', 'target/out_pressure', epoch=epoch+1, batch_idx=k+1)
                _log_tensor_if_not_finite(time_steps, finite_log_path, 'test', 'time_steps', epoch=epoch+1, batch_idx=k+1)

                model_out = model(in_press, time_steps)
                _log_tensor_if_not_finite(model_out, finite_log_path, 'test', 'model_out', epoch=epoch+1, batch_idx=k+1)
                loss_value = _safe_mse(model_out, out_pressure)  # 测试也用安全MSE
                _log_tensor_if_not_finite(loss_value, finite_log_path, 'test', 'loss_value', epoch=epoch+1, batch_idx=k+1)
                total_test_loss += loss_value.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_loss_history.append(avg_test_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

        with open(loss_log_path, 'a') as log_file:
            log_file.write(f"{epoch + 1}, {avg_train_loss:.6f}, {avg_valid_loss:.6f}, {avg_test_loss:.6f}\n")

        # ========== 早停逻辑 ==========
        if enable_early_stopping:
            # 根据监控指标选择当前值
            if monitor == "train_loss":
                current_metric = avg_train_loss
            elif monitor == "val_loss":
                current_metric = avg_valid_loss
            elif monitor == "test_loss":
                current_metric = avg_test_loss
            else:
                current_metric = avg_valid_loss  # 默认监控验证损失
            
            # 判断是否有改善
            if is_better(current_metric, best_metric):
                best_metric = current_metric
                patience_counter = 0
                # 保存最佳模型权重
                if restore_best_weights:
                    best_model_state = model.state_dict().copy()
                
                # 使用配置文件中的模型保存路径
                if cfg and 'training' in cfg and 'model_save_path' in cfg['training']:
                    model_save_path = cfg['training']['model_save_path']
                    # 确保目录存在
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Model saved to: {model_save_path} (Best {monitor}: {current_metric:.6f})")
                else:
                    # 回退到原来的保存方式
                    torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_{attention_type}.pt"))
                    print(f"Model saved (Best {monitor}: {current_metric:.6f})")
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{patience} (current {monitor}: {current_metric:.6f}, best: {best_metric:.6f})")
            
            # 检查是否触发早停
            if patience_counter >= patience:
                print("Early stopping triggered!")
                # 恢复最佳权重
                if restore_best_weights and best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("Restored best model weights")
                break
        else:
            # 不启用早停时的传统逻辑
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                
                # 使用配置文件中的模型保存路径
                if cfg and 'training' in cfg and 'model_save_path' in cfg['training']:
                    model_save_path = cfg['training']['model_save_path']
                    # 确保目录存在
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Model saved to: {model_save_path} (Best Model Updated)")
                else:
                    # 回退到原来的保存方式
                    torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_{attention_type}.pt"))
                    print("Model saved (Best Model Updated)")
        
        # ========== 保存断点 ==========
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss_history,
            'valid_loss_history': valid_loss_history,
            'test_loss_history': test_loss_history,
            'best_valid_loss': best_valid_loss,
            'best_metric': best_metric,
            'patience_counter': patience_counter,
            'best_model_state': best_model_state
        }
        torch.save(checkpoint, checkpoint_path)
        
        # ========== 学习率调度器步进 ==========
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Scheduler stepped: current lr={current_lr:.8f}")

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

                        # 动态计算形状，支持不同尺寸的数据
                        input_size = sample_input[0].numel()
                        output_size = sample_output[0].numel()
                        
                        # 计算输入数据的最佳形状（尽量接近正方形）
                        input_dim = int(input_size ** 0.5)
                        if input_dim * input_dim == input_size:
                            input_shape = (input_dim, input_dim)
                        else:
                            # 寻找最接近的因子对
                            factors = []
                            for i in range(1, int(input_size ** 0.5) + 1):
                                if input_size % i == 0:
                                    factors.append((i, input_size // i))
                            if factors:
                                input_shape = min(factors, key=lambda x: abs(x[0] - x[1]))
                            else:
                                input_shape = (1, input_size)
                        
                        # 计算输出数据的最佳形状
                        output_dim = int(output_size ** 0.5)
                        if output_dim * output_dim == output_size:
                            output_shape = (output_dim, output_dim)
                        else:
                            # 寻找最接近的因子对
                            factors = []
                            for i in range(1, int(output_size ** 0.5) + 1):
                                if output_size % i == 0:
                                    factors.append((i, output_size // i))
                            if factors:
                                output_shape = min(factors, key=lambda x: abs(x[0] - x[1]))
                            else:
                                output_shape = (1, output_size)
                        
                        input_pressure = sample_input[0].view(*input_shape).cpu().numpy()
                        true_pressure = sample_output[0].view(*output_shape).cpu().numpy()
                        predicted_pressure = predictions[0].view(*output_shape).cpu().numpy()

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
                    print("Validation set is insufficient; cannot generate visualization.")

        if (epoch + 1) % vis_interval == 0:
            plot_dir = os.path.join(save_dir, "loss_plots")
            os.makedirs(plot_dir, exist_ok=True)
            loss_fig_path = os.path.join(plot_dir, f"loss_curve_epoch_{epoch + 1}.png")
            plot_losses(train_loss_history, valid_loss_history, test_loss_history, save_path=loss_fig_path)

    plt.ioff()
    return model, train_loss_history, valid_loss_history, test_loss_history


def _investigate_nonfinite_with_hooks(model, inputs, finite_log_path, limit=10):
    """
    当检测到非有限输出时，对模型各层注册 forward-hook，定位首先产生 NaN/Inf 的模块。
    inputs: tuple/list of tensors, 将以 model(*inputs) 的形式重新前向一次
    """
    try:
        module_name_map = {}
        handles = []
        logged = set()
        counter = {'n': 0}

        for name, m in model.named_modules():
            if name == '':
                continue
            module_name_map[m] = name
            def hook_fn(module, inp, out):
                if counter['n'] >= limit:
                    return
                def check_tensor(t, tag_suffix=''):
                    try:
                        if isinstance(t, torch.Tensor):
                            if not torch.isfinite(t.detach()).all():
                                mod_name = module_name_map.get(module, module.__class__.__name__)
                                tag = f"module::{mod_name}{tag_suffix}"
                                if tag not in logged:
                                    _append_finite_record(finite_log_path, 'hook', tag, t)
                                    logged.add(tag)
                                    counter['n'] += 1
                    except Exception:
                        pass
                # 处理输出可能为 Tensor / tuple / list
                if isinstance(out, torch.Tensor):
                    check_tensor(out)
                elif isinstance(out, (list, tuple)):
                    for idx, item in enumerate(out):
                        check_tensor(item, tag_suffix=f"/out[{idx}]")
                else:
                    # 不支持的输出类型，忽略
                    pass
            handles.append(m.register_forward_hook(hook_fn))

        with torch.no_grad():
            if isinstance(inputs, (list, tuple)):
                model(*inputs)
            else:
                model(inputs)
    except Exception as e:
        print(f"[finite_debug] hook investigation failed: {e}")
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


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
        # 确保父目录存在（用于写入 finite_debug.txt 等文件）
        os.makedirs(parent_dir, exist_ok=True)
        loss_log_dir = os.path.join(parent_dir, "loss_logs")
        finite_log_path = os.path.join(parent_dir, "finite_debug.txt")
    else:
        base_dir = f"attention_results/{attention_type}"
        loss_log_dir = os.path.join(base_dir, "loss_logs")
        finite_log_path = os.path.join(base_dir, "finite_debug.txt")
    os.makedirs(loss_log_dir, exist_ok=True)
    loss_log_path = os.path.join(loss_log_dir, "test_loss_log.txt")

    # 测试阶段也写入一次启动记录，确保文件存在
    try:
        with open(finite_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'event': 'test_run_start', 'attention_type': attention_type}) + '\n')
    except Exception as e:
        print(f"[finite_debug] failed to init test log {finite_log_path}: {e}")

    # ===== 检查模型参数/缓冲区是否存在非有限值 =====
    try:
        for name, param in model.named_parameters():
            if param is None:
                continue
            if not torch.isfinite(param.detach()).all():
                _append_finite_record(finite_log_path, 'test_param', f'param::{name}', param)
        # 同时检查 buffers（例如 BatchNorm 的 running_mean/var）
        for name, buf in model.named_buffers():
            if buf is None:
                continue
            if not torch.isfinite(buf.detach()).all():
                _append_finite_record(finite_log_path, 'test_buffer', f'buffer::{name}', buf)
    except Exception as e:
        print(f"[finite_debug] param/buffer check failed: {e}")

    with open(loss_log_path, 'w') as log_file:
        log_file.write("Batch, Test Loss\n")

    with torch.no_grad():
        for idx, (in_press, out_pressure, time_steps) in enumerate(test_loader):
            # CPU数据加载优化：测试函数中也使用非阻塞传输
            if device.type == 'cuda':
                in_press = in_press.to(device, non_blocking=True)
                out_pressure = out_pressure.to(device, non_blocking=True)
                time_steps = time_steps.to(device, non_blocking=True)
            else:
                in_press, out_pressure, time_steps = (
                    in_press.to(device), out_pressure.to(device), time_steps.to(device)
                )

            # 输入 finite 检查
            _log_tensor_if_not_finite(in_press, finite_log_path, 'test', 'input/in_press', batch_idx=idx+1)
            _log_tensor_if_not_finite(out_pressure, finite_log_path, 'test', 'target/out_pressure', batch_idx=idx+1)
            _log_tensor_if_not_finite(time_steps, finite_log_path, 'test', 'time_steps', batch_idx=idx+1)

            model_out = model(in_press, time_steps)
            # 若输出非有限，触发 hooks 调查
            try:
                if isinstance(model_out, torch.Tensor) and (not torch.isfinite(model_out.detach()).all()):
                    _investigate_nonfinite_with_hooks(model, (in_press, time_steps), finite_log_path, limit=10)
                    # 专门对 CNNResizeBaselineModel 做逐步检查
                    if hasattr(model, 'backbone') and hasattr(model, 'head') and hasattr(model, 'H_in') and hasattr(model, 'H_out'):
                        _inspect_forward_cnn_resize(model, (in_press, time_steps), finite_log_path, batch_idx=idx+1)
            except Exception as e:
                print(f"[finite_debug] model_out finite check failed: {e}")

            _log_tensor_if_not_finite(model_out, finite_log_path, 'test_fn', 'model_out', batch_idx=idx+1)
            loss_value = _safe_mse(model_out, out_pressure)  # 测试也用安全MSE
            _log_tensor_if_not_finite(loss_value, finite_log_path, 'test_fn', 'loss_value', batch_idx=idx+1)
            total_test_loss += loss_value.item()

    avg_test_loss = total_test_loss / len(test_loader)
    with open(loss_log_path, 'a') as log_file:
        log_file.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    return avg_test_loss

    with open(loss_log_path, 'a') as log_file:
        log_file.write(f"{idx + 1}, {loss_value.item():.6f}\n")

    if vis_enabled and idx < max_samples:
        # 动态计算形状，支持不同尺寸的数据
        input_size = in_press[0].numel()
        output_size = out_pressure[0].numel()
        
        # 计算输入数据的最佳形状（尽量接近正方形）
        input_dim = int(input_size ** 0.5)
        if input_dim * input_dim == input_size:
            input_shape = (input_dim, input_dim)
        else:
            # 寻找最接近的因子对
            factors = []
            for i in range(1, int(input_size ** 0.5) + 1):
                if input_size % i == 0:
                    factors.append((i, input_size // i))
            if factors:
                input_shape = min(factors, key=lambda x: abs(x[0] - x[1]))
            else:
                input_shape = (1, input_size)
        
        # 计算输出数据的最佳形状
        output_dim = int(output_size ** 0.5)
        if output_dim * output_dim == output_size:
            output_shape = (output_dim, output_dim)
        else:
            # 寻找最接近的因子对
            factors = []
            for i in range(1, int(output_size ** 0.5) + 1):
                if output_size % i == 0:
                    factors.append((i, output_size // i))
            if factors:
                output_shape = min(factors, key=lambda x: abs(x[0] - x[1]))
            else:
                output_shape = (1, output_size)
        
        input_pressure = in_press[0].view(*input_shape).cpu().numpy()
        true_pressure = out_pressure[0].view(*output_shape).cpu().numpy()
        predicted_pressure = model_out[0].view(*output_shape).cpu().numpy()

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
    print(f"Test completed, {attention_type} Test Loss: {avg_test_loss:.6f}")

    with open(loss_log_path, 'a') as log_file:
        log_file.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    return avg_test_loss


def _safe_mse(pred, target):
    try:
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
        diff = pred - target
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)
        return (diff * diff).mean()
    except Exception:
        # 回退：尽量不因异常中断
        return ((pred - target) ** 2).mean()