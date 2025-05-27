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

        # 新增：自动读取yaml中的掩码路径和权重
        self.mask_type = self.cfg['training'].get('mask_type', 'svd')
        self.svd_mask_paths = self.cfg['training'].get('svd_mask_paths', [])
        self.svd_mask_weights = self.cfg['training'].get('svd_mask_weights', [])
        if not self.svd_mask_weights or len(self.svd_mask_weights) != len(self.svd_mask_paths):
            # 默认等权
            self.svd_mask_weights = [1.0 / len(self.svd_mask_paths)] * len(self.svd_mask_paths)

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

        # ==== 自动化多掩码加权融合 ====
        if self.mask_type == 'svd' and self.svd_mask_paths:
            mask_sum = torch.zeros(self.output_dim)
            mask_found = False
            for path, weight in zip(self.svd_mask_paths, self.svd_mask_weights):
                if os.path.exists(path):
                    svd_mask = torch.load(path).float().flatten()
                    if svd_mask.shape[0] != self.output_dim:
                        svd_mask = svd_mask.view(-1)
                    mask_sum += svd_mask * weight
                    mask_found = True
            if mask_found and mask_sum.max() > 0:
                mask = mask_sum / mask_sum.max()
            else:
                mask = torch.ones(self.output_dim)
        else:
            mask = region_mask

        return in_press_flat, pressure_flat, time_step, mask, reynolds_idx
