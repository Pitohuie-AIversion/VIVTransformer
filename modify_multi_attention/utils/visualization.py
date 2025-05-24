import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def robust_np(x):
    # 自动转为 numpy 并 squeeze
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.squeeze(x)

def plot_losses(train_loss, valid_loss, test_loss, save_path=None, logscale=True):
    if len(train_loss) == 0 or len(valid_loss) == 0 or len(test_loss) == 0:
        print("⚠️ 损失列表为空，无法绘制Loss曲线！")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.title('Loss Curve' + (' (Log Scale)' if logscale else ''))
    plt.legend()
    if logscale:
        plt.yscale('log')
        plt.ylabel('Loss (Log Scale)')
    else:
        plt.ylabel('Loss')
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Loss曲线已保存到: {save_path}")
    else:
        plt.show()
    plt.close('all')

def plot_comparison_figure(input_pressure, true_pressure, predicted_pressure, reynolds_number, time_step, epoch, idx, attention_type, parent_dir="attention_results", mode="test"):
    result_dir = os.path.join(parent_dir, attention_type, "visualization_results")
    os.makedirs(result_dir, exist_ok=True)
    input_pressure = robust_np(input_pressure)
    true_pressure = robust_np(true_pressure)
    predicted_pressure = robust_np(predicted_pressure)

    vmax = max(np.abs(input_pressure).max(), np.abs(true_pressure).max(), np.abs(predicted_pressure).max())
    plt.figure(figsize=(18, 5))
    for i, (data, title) in enumerate(zip(
        [input_pressure, true_pressure, predicted_pressure],
        [f"Input Pressure Matrix at t={time_step:.2f}", f"True Pressure Matrix at t={time_step:.2f}", f"Predicted Pressure Matrix at t={time_step:.2f}"]
    )):
        plt.subplot(1, 3, i+1)
        im = plt.imshow(data, cmap='coolwarm', interpolation='nearest', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(result_dir, f"{mode}_epoch_{epoch}_sample_{idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close('all')

def plot_difference_figure(true_pressure, predicted_pressure, reynolds_number, time_step, epoch, idx, attention_type, parent_dir="attention_results", mode="test"):
    result_dir = os.path.join(parent_dir, attention_type, "difference_results")
    os.makedirs(result_dir, exist_ok=True)
    true_pressure = robust_np(true_pressure)
    predicted_pressure = robust_np(predicted_pressure)
    difference = np.abs(true_pressure - predicted_pressure)
    save_prefix = f"Re_{reynolds_number}_time_{time_step:.2f}_{mode}_epoch_{epoch}_sample_{idx}"
    npy_save_path = os.path.join(result_dir, f"{save_prefix}_difference_matrix.npy")
    np.save(npy_save_path, difference)
    csv_save_path = os.path.join(result_dir, f"{save_prefix}_difference_matrix.csv")
    np.savetxt(csv_save_path, difference, delimiter=",")
    plt.figure(figsize=(6, 5))
    vmax = np.abs(difference).max()
    im = plt.imshow(difference, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Difference (|True - Predicted|) at t={time_step:.2f}")
    plt.tight_layout()
    save_path = os.path.join(result_dir, f"{save_prefix}_difference.png")
    plt.savefig(save_path, dpi=300)
    plt.close('all')

def ensure_2d(array, h=None, w=None, name="输入"):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.squeeze(array)
    if array.ndim == 1:
        if h is not None and w is not None:
            assert array.shape[0] == h * w, f"{name}长度与指定h*w不一致，当前len={array.shape[0]}，h={h}, w={w}"
            array = array.reshape(h, w)
        else:
            N = array.shape[0]
            side = int(np.sqrt(N))
            assert side * side == N, f"{name}长度不是完全平方数，无法自动reshape成二维"
            array = array.reshape(side, side)
    assert array.ndim == 2, f"{name}必须为2维，当前shape={array.shape}"
    return array

def plot_svd_spectrum(tensor2d, label='Pred', color='r', linestyle='-'):
    tensor2d = ensure_2d(tensor2d, name="SVD谱输入")
    U, S, Vh = np.linalg.svd(tensor2d, full_matrices=False)
    plt.plot(S, marker='o', linestyle=linestyle, color=color, label=label)
    return S

def compare_svd_spectra(pred, target, h, w, save_path=None, pred_label="Pred", tgt_label="True", extra_spectra=None):
    pred = ensure_2d(pred, h, w, "Pred SVD输入")
    target = ensure_2d(target, h, w, "True SVD输入")
    plt.figure(figsize=(8, 5))
    plot_svd_spectrum(target, label=tgt_label, color='b')
    plot_svd_spectrum(pred, label=pred_label, color='r')
    if extra_spectra is not None:
        for arr, label, color, ls in extra_spectra:
            arr = ensure_2d(arr, h, w, label)
            plot_svd_spectrum(arr, label=label, color=color, linestyle=ls)
    plt.yscale('log')
    plt.xlabel('SVD mode index')
    plt.ylabel('Singular value (log scale)')
    plt.legend()
    plt.title('SVD Spectrum Comparison')
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"SVD能量谱图已保存到: {save_path}")
    else:
        plt.show()
    plt.close('all')

def compare_svd_experiments(pred_nosvd, pred_withsvd, target, h=200, w=200, save_path=None):
    pred_nosvd = ensure_2d(pred_nosvd, h, w, "pred_nosvd")
    pred_withsvd = ensure_2d(pred_withsvd, h, w, "pred_withsvd")
    target = ensure_2d(target, h, w, "target")
    extra = [
        (pred_withsvd, "Pred w/ SVD Loss", "g", "--"),
    ]
    compare_svd_spectra(pred_nosvd, target, h, w, save_path=save_path, pred_label="Pred w/o SVD Loss", extra_spectra=extra)

def plot_svd_modes(tensor2d, n_modes=3, title_prefix="Field", vmax=None, save_path=None):
    tensor2d = ensure_2d(tensor2d, name=title_prefix)
    U, S, Vh = np.linalg.svd(tensor2d, full_matrices=False)
    vmax = vmax or np.abs(tensor2d).max()
    fig, axs = plt.subplots(1, n_modes, figsize=(4*n_modes, 4))
    for i in range(n_modes):
        mode_img = S[i] * np.outer(U[:,i], Vh[i,:])
        im = axs[i].imshow(mode_img, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
        axs[i].set_title(f"{title_prefix} - Mode {i+1}\nσ={S[i]:.2e}")
        axs[i].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"SVD分解模态图已保存到：{save_path}")
    else:
        plt.show()
    plt.close('all')
    return [S[i] * np.outer(U[:,i], Vh[i,:]) for i in range(n_modes)]

def plot_svd_modes_comparison(pred_nosvd, pred_withsvd, target, h, w, n_modes=3, save_dir=None):
    pred_nosvd = ensure_2d(pred_nosvd, h, w, "pred_nosvd")
    pred_withsvd = ensure_2d(pred_withsvd, h, w, "pred_withsvd")
    target = ensure_2d(target, h, w, "target")
    vmax = max(np.abs(target).max(), np.abs(pred_nosvd).max(), np.abs(pred_withsvd).max())
    plot_svd_modes(target, n_modes=n_modes, title_prefix="True", vmax=vmax,
                   save_path=os.path.join(save_dir, "svd_modes_true.png") if save_dir else None)
    plot_svd_modes(pred_nosvd, n_modes=n_modes, title_prefix="Pred w/o SVD", vmax=vmax,
                   save_path=os.path.join(save_dir, "svd_modes_nosvd.png") if save_dir else None)
    plot_svd_modes(pred_withsvd, n_modes=n_modes, title_prefix="Pred w/ SVD", vmax=vmax,
                   save_path=os.path.join(save_dir, "svd_modes_withsvd.png") if save_dir else None)

def plot_svd_mode_grid_comparison(pred_nosvd, pred_withsvd, target, h, w, n_modes=3, save_path=None):
    pred_nosvd = ensure_2d(pred_nosvd, h, w, "pred_nosvd")
    pred_withsvd = ensure_2d(pred_withsvd, h, w, "pred_withsvd")
    target = ensure_2d(target, h, w, "target")
    arrs = [
        ('True', target),
        ('Pred w/o SVD', pred_nosvd),
        ('Pred w/ SVD', pred_withsvd),
    ]
    vmax = max(np.abs(arrs[0][1]).max(), np.abs(arrs[1][1]).max(), np.abs(arrs[2][1]).max())
    fig, axs = plt.subplots(len(arrs), n_modes, figsize=(4*n_modes, 4*len(arrs)))
    for row, (label, data) in enumerate(arrs):
        U, S, Vh = np.linalg.svd(data, full_matrices=False)
        for col in range(n_modes):
            mode_img = S[col] * np.outer(U[:,col], Vh[col,:])
            im = axs[row, col].imshow(mode_img, cmap='coolwarm', vmin=-vmax, vmax=vmax)
            fig.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
            axs[row, col].set_title(f"{label} - Mode {col+1}\nσ={S[col]:.2e}")
            axs[row, col].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"分解模态空间对比图已保存到：{save_path}")
    else:
        plt.show()
    plt.close('all')

# ======= SVD重构及曲线 =======
def svd_n_mode_reconstruction(field, n_modes=10):
    field = robust_np(field)
    U, S, Vh = np.linalg.svd(field, full_matrices=False)
    H, W = field.shape
    recon = np.zeros((H, W))
    for i in range(n_modes):
        recon += S[i] * np.outer(U[:, i], Vh[i, :])
    return recon, S

def svd_energy_curve(field, max_modes=10):
    field = robust_np(field)
    U, S, Vh = np.linalg.svd(field, full_matrices=False)
    total_energy = np.sum(S)
    energies = [np.sum(S[:i+1]) / total_energy for i in range(max_modes)]
    return energies, S

def svd_reconstruction_mse_curve(field, max_modes=10):
    field = robust_np(field)
    U, S, Vh = np.linalg.svd(field, full_matrices=False)
    H, W = field.shape
    mses = []
    for n in range(1, max_modes+1):
        recon = np.zeros((H, W))
        for i in range(n):
            recon += S[i] * np.outer(U[:, i], Vh[i, :])
        mse = np.mean((field - recon) ** 2)
        mses.append(mse)
    return mses

def plot_top_n_svd_modes(field, n_modes=10, save_path=None, vmax=None):
    field = robust_np(field)
    U, S, Vh = np.linalg.svd(field, full_matrices=False)
    H, W = field.shape
    if vmax is None:
        vmax = np.abs(field).max()
    fig, axs = plt.subplots(2, n_modes, figsize=(3*n_modes, 6))
    # 第一行：每一阶单独模态
    for i in range(n_modes):
        mode_img = S[i] * np.outer(U[:,i], Vh[i,:])
        im = axs[0, i].imshow(mode_img, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=axs[0, i], fraction=0.046, pad=0.04)
        axs[0, i].set_title(f"Mode {i+1}\nσ={S[i]:.2e}")
        axs[0, i].axis('off')
    # 第二行：每一阶累加重构
    recon = np.zeros((H, W))
    for i in range(n_modes):
        recon += S[i] * np.outer(U[:,i], Vh[i,:])
        im = axs[1, i].imshow(recon, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=axs[1, i], fraction=0.046, pad=0.04)
        mse = np.mean((field - recon)**2)
        axs[1, i].set_title(f"Sum 1~{i+1}\nMSE={mse:.2e}")
        axs[1, i].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"前{n_modes}阶SVD模态分解图已保存到：{save_path}")
    else:
        plt.show()
    plt.close('all')

def plot_energy_and_mse_curves(field, max_modes=10, save_path=None):
    energies, S = svd_energy_curve(field, max_modes)
    mses = svd_reconstruction_mse_curve(field, max_modes)
    plt.figure(figsize=(8,4))
    plt.plot(range(1, max_modes+1), energies, marker='o', label='Cumulative Energy')
    plt.plot(range(1, max_modes+1), mses, marker='s', label='Reconstruction MSE')
    plt.xlabel("Number of modes")
    plt.ylabel("Ratio / MSE")
    plt.legend()
    plt.grid(True)
    plt.title("SVD能量累积与还原误差曲线")
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"SVD能量/MSE递减曲线已保存到：{save_path}")
    else:
        plt.show()
    plt.close('all')
