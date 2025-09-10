import torch
import numpy as np
from torch.utils.data import Dataset


class PressureDataset(Dataset):
    def __init__(self, merged_file_path):
        self.data = torch.load(merged_file_path)
        self.in_pressures = self.data['in_pressure']
        self.pressures = self.data['pressure']
        self.time_steps = np.array(self.data['time_steps'])

        if len(self.time_steps.shape) == 1:
            self.time_steps = np.array([self.time_steps] * len(self.in_pressures))

        self.num_samples = len(self.in_pressures) * len(self.in_pressures[0])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        reynolds_idx = idx // len(self.in_pressures[0])
        time_step_idx = idx % len(self.in_pressures[0])

        in_press_flat = self.in_pressures[reynolds_idx, time_step_idx].view(-1)
        pressure_flat = self.pressures[reynolds_idx, time_step_idx].view(-1)
        # 数据净化：将 NaN -> 0，Inf -> 有界值，避免后续产生 NaN/Inf 损失
        in_press_flat = torch.nan_to_num(in_press_flat, nan=0.0, posinf=1e6, neginf=-1e6)
        pressure_flat = torch.nan_to_num(pressure_flat, nan=0.0, posinf=1e6, neginf=-1e6)

        time_step = self.time_steps[reynolds_idx][time_step_idx]

        return in_press_flat, pressure_flat, time_step
