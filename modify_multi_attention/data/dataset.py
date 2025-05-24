import torch
import numpy as np
from torch.utils.data import Dataset
import yaml

class PressureDataset(Dataset):
    def __init__(self, merged_file_path, config_path='modify_multi_attention/configs/config.yaml'):
        self.data = torch.load(merged_file_path)
        self.in_pressures = self.data['in_pressure']   # [N_re, N_t, 20, 20]
        self.pressures = self.data['pressure']         # [N_re, N_t, 200, 200]
        self.time_steps = np.array(self.data['time_steps'])  # [N_re, N_t] or [N_t]

        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        # 自动扩展 time_steps 至 [N_re, N_t]
        if self.time_steps.ndim == 1:
            # 如果只有 [N_t]，扩展为 [N_re, N_t]
            self.time_steps = np.tile(self.time_steps, (self.in_pressures.shape[0], 1))
        elif self.time_steps.ndim == 2:
            assert self.time_steps.shape[0] == self.in_pressures.shape[0], "雷诺数数量不一致"
            assert self.time_steps.shape[1] == self.in_pressures.shape[1], "时间步数量不一致"
        else:
            raise ValueError(f"time_steps shape 不支持: {self.time_steps.shape}")

        self.num_samples = self.in_pressures.shape[0] * self.in_pressures.shape[1]
        self.input_dim = self.in_pressures.shape[2] * self.in_pressures.shape[3]
        self.output_dim = self.pressures.shape[2] * self.pressures.shape[3]

        self.mask_regions = self.cfg.get('training', {}).get('mask_regions', [])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        N_t = self.in_pressures.shape[1]
        reynolds_idx = idx // N_t
        time_step_idx = idx % N_t

        try:
            in_press = self.in_pressures[reynolds_idx, time_step_idx]
            pressure = self.pressures[reynolds_idx, time_step_idx]
            in_press_flat = torch.as_tensor(in_press).float().reshape(-1)
            pressure_flat = torch.as_tensor(pressure).float().reshape(-1)
            time_step = float(self.time_steps[reynolds_idx][time_step_idx])
        except Exception as e:
            print(f"[Dataset Error] idx={idx}, reynolds_idx={reynolds_idx}, time_step_idx={time_step_idx}, err={e}")
            return None  # collate_fn会自动过滤

        output_side = int(self.output_dim ** 0.5)
        mask = torch.zeros(self.output_dim, dtype=torch.float32)
        if not self.mask_regions:
            mask[:] = 1.0  # 无掩码配置，默认全1
        else:
            for x_start, x_end, y_start, y_end in self.mask_regions:
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        index = y * output_side + x
                        if 0 <= index < self.output_dim:
                            mask[index] = 1.0

        return in_press_flat, pressure_flat, torch.tensor(time_step, dtype=torch.float32), mask, torch.tensor(reynolds_idx, dtype=torch.long)
