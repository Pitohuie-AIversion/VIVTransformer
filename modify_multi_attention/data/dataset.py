# File: data/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
import yaml

class PressureDataset(Dataset):
    def __init__(self, merged_file_path, config_path='modify_multi_attention/configs/config.yaml'):
        # —— 原有数据加载 ——
        self.data = torch.load(merged_file_path)
        self.in_pressures = self.data['in_pressure']   # shape [num_re, num_t, ...]
        self.pressures    = self.data['pressure']      # shape [num_re, num_t, H, W]
        self.time_steps   = np.array(self.data['time_steps'])
        if self.time_steps.ndim == 1:
            self.time_steps = np.array([self.time_steps] * len(self.in_pressures))

        # 样本总数
        self.num_samples = len(self.in_pressures) * len(self.in_pressures[0])

        # —— 计算 input_dim 和 output_dim ——
        ip_shape = self.in_pressures.shape
        if len(ip_shape) == 5:
            _, _, C, H, W = ip_shape
            self.input_dim = C * H * W
        elif len(ip_shape) == 4:
            _, _, H, W = ip_shape
            self.input_dim = H * W
        else:
            raise ValueError(f"Unexpected in_pressures shape: {ip_shape}")

        pr_shape = self.pressures.shape
        if len(pr_shape) == 4:
            _, _, H2, W2 = pr_shape
            self.output_dim = H2 * W2
        else:
            raise ValueError(f"Unexpected pressures shape: {pr_shape}")

        # —— 加载配置 ——
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        # 二值掩码区域
        self.mask_regions = self.cfg['training'].get('mask_regions', [])

        # —— 多模态 SVD 权重图加载 ——
        if self.cfg['training'].get('mask_weight_modes_enabled', False):
            paths = self.cfg['training']['mask_weight_mode_paths']
            self.mode_weights = [torch.load(p) for p in paths]
            self.num_modes    = len(self.mode_weights)
            self.svd_iter_idx = self.cfg['training'].get('mask_svd_iter_idx', 0)

            # —— 动态检测是否有 iter 维度 ——
            first_mode = self.mode_weights[0]
            if isinstance(first_mode, list) and first_mode:
                first_time_entry = first_mode[0]
                if isinstance(first_time_entry, list):
                    self.has_iter_dim = True
                    self.num_iters    = len(first_time_entry)
                else:
                    self.has_iter_dim = False
                    self.num_iters    = 1
            else:
                self.has_iter_dim = False
                self.num_iters    = 1

            # clamp svd_iter_idx，避免越界
            if self.svd_iter_idx >= self.num_iters:
                self.svd_iter_idx = self.num_iters - 1

            # —— 验证 mode_weights 与 in_pressures 尺寸对齐，否则禁用多模态加权 ——
            expected_re = self.in_pressures.shape[0]
            expected_t  = self.in_pressures.shape[1]
            mismatch = False
            for m_w in self.mode_weights:
                if len(m_w) != expected_re:
                    mismatch = True
                    break
                for time_list in m_w:
                    if len(time_list) != expected_t:
                        mismatch = True
                        break
                if mismatch:
                    break
            if mismatch:
                print("⚠️ Warning: SVD 权重维度与数据不匹配，已禁用多模态加权")
                self.mode_weights = None
                self.num_modes    = 0
                self.has_iter_dim = False
                self.num_iters    = 1
        else:
            self.mode_weights = None
            self.num_modes    = 0
            self.has_iter_dim = False
            self.num_iters    = 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 计算 re_idx 与 time_step_idx
        num_t         = len(self.in_pressures[0])
        reynolds_idx  = idx // num_t
        time_step_idx = idx % num_t

        # 取出并展平输入与目标
        in_press_flat = self.in_pressures[reynolds_idx, time_step_idx].view(-1)
        pressure_flat = self.pressures   [reynolds_idx, time_step_idx].view(-1)

        # 构造二值掩码
        mask = torch.zeros(self.output_dim, dtype=torch.float)
        side = int(self.output_dim ** 0.5)
        for x0, x1, y0, y1 in self.mask_regions:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    ix = y * side + x
                    if ix < self.output_dim:
                        mask[ix] = 1.0

        # —— 构造多模态权重 maps ——
        if self.mode_weights is not None and self.num_modes > 0:
            wm_list = []
            for m in range(self.num_modes):
                # 防越界检测
                if reynolds_idx >= len(self.mode_weights[m]) or \
                   time_step_idx >= len(self.mode_weights[m][0]):
                    print(f"⚠️ mode {m} 索引越界，用 0 填充")
                    w2d = torch.zeros(self.output_dim, dtype=torch.float)
                else:
                    entry = self.mode_weights[m][reynolds_idx][time_step_idx]
                    if entry is None:
                        w2d = torch.zeros(self.output_dim, dtype=torch.float)
                    else:
                        if self.has_iter_dim:
                            sub = entry[self.svd_iter_idx] if self.svd_iter_idx < len(entry) else None
                            w2d = sub if sub is not None else torch.zeros(self.output_dim, dtype=torch.float)
                        else:
                            w2d = entry
                wm_list.append(w2d.view(-1))
            weight_maps = torch.stack(wm_list, dim=0)  # (num_modes, output_dim)
        else:
            weight_maps = mask.unsqueeze(0)  # (1, output_dim)

        # 返回五元组：in_press, target, mask, weight_maps, reynolds_idx
        return in_press_flat, pressure_flat, mask, weight_maps, reynolds_idx
