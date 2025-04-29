import torch
import numpy as np
from torch.utils.data import Dataset
import yaml

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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        reynolds_idx = idx // len(self.in_pressures[0])
        time_step_idx = idx % len(self.in_pressures[0])

        # 获取展平的输入和输出
        in_press_flat = self.in_pressures[reynolds_idx, time_step_idx].view(-1)
        pressure_flat = self.pressures[reynolds_idx, time_step_idx].view(-1)
        time_step = self.time_steps[reynolds_idx][time_step_idx]

        output_side = int(self.input_dim ** 0.5)
        mask = torch.zeros(self.output_dim).float()

        for x_start, x_end, y_start, y_end in self.mask_regions:
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    index = y * output_side + x
                    if index < self.output_dim:
                        mask[index] = 1.0

        return in_press_flat, pressure_flat, time_step, mask, reynolds_idx
