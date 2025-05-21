import torch
import numpy as np
from torch.utils.data import Dataset
import yaml
import os

class PressureDataset(Dataset):
    def __init__(self, merged_file_path, config_path='modify_multi_attention/configs/config.yaml'):
        self.data = torch.load(merged_file_path)
        self.in_pressures = self.data['in_pressure']
        self.pressures = self.data['pressure']
        self.time_steps = np.array(self.data['time_steps'])

        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        if len(self.time_steps.shape) == 1:
            self.time_steps = np.array([self.time_steps] * len(self.in_pressures))

        self.num_samples = len(self.in_pressures) * len(self.in_pressures[0])
        self.input_dim = self.in_pressures[0, 0].numel()
        self.output_dim = self.pressures[0, 0].numel()
        self.mask_regions = self.cfg['training'].get('mask_regions', [])

        # 新增：读取SVD主模态掩码相关配置
        self.mask_type = self.cfg['training'].get('mask_type', 'region')   # "region" 或 "svd"
        self.svd_mask_dir = self.cfg['training'].get('svd_mask_dir', None)
        self.svd_mask_mode = self.cfg['training'].get('svd_mask_mode', 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        reynolds_idx = idx // len(self.in_pressures[0])
        time_step_idx = idx % len(self.in_pressures[0])

        in_press_flat = self.in_pressures[reynolds_idx, time_step_idx].view(-1)
        pressure_flat = self.pressures[reynolds_idx, time_step_idx].view(-1)
        time_step = self.time_steps[reynolds_idx][time_step_idx]

        output_side = int(self.input_dim ** 0.5)
        region_mask = torch.zeros(self.output_dim).float()
        for x_start, x_end, y_start, y_end in self.mask_regions:
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    index = y * output_side + x
                    if index < self.output_dim:
                        region_mask[index] = 1.0

        # ==== SVD主模态掩码自动融合 ====
        if self.mask_type == 'svd' and self.svd_mask_dir is not None:
            mask_dir = os.path.join(self.svd_mask_dir, f"mode{self.svd_mask_mode}")
            # 文件命名务必与你SVD批量保存一致！
            mask_name = f"Re_{reynolds_idx}_time_{time_step:.2f}_train_epoch_0_sample_{time_step_idx}_difference_matrix.pt"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                svd_mask = torch.load(mask_path).float().flatten()
                # 确保维度一致
                if svd_mask.shape[0] != self.output_dim:
                    svd_mask = svd_mask.view(-1)
                mask = svd_mask
            else:
                mask = torch.ones(self.output_dim)  # 若未找到则不加权
        else:
            mask = region_mask

        return in_press_flat, pressure_flat, time_step, mask, reynolds_idx
