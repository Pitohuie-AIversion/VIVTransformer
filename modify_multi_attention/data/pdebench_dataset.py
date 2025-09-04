import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PDEBenchDataset(Dataset):
    """
    PDEBench数据集类，支持HDF5格式的数据加载
    
    Args:
        data_path: HDF5文件路径或包含HDF5文件的目录路径
        transform: 数据变换函数（可选）
        max_samples: 最大样本数量限制（可选）
    """
    
    def __init__(self, data_path, transform=None, max_samples=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.max_samples = max_samples
        
        # 存储数据信息
        self.data_info = {}
        self.sample_indices = []
        self.configs = []
        
        self._load_data_info()
        
    def _load_data_info(self):
        """加载数据文件信息"""
        # 检查路径是否存在
        if not self.data_path.exists():
            raise ValueError(f"数据路径不存在: {self.data_path}")
            
        if self.data_path.is_file():
            # 检查文件扩展名
            if self.data_path.suffix.lower() in ['.h5', '.hdf5']:
                self._process_single_file(self.data_path)
            else:
                raise ValueError(f"不支持的文件格式: {self.data_path.suffix}，仅支持.h5和.hdf5文件")
        elif self.data_path.is_dir():
            # 目录中的多个HDF5文件
            h5_files = list(self.data_path.glob("*.h5")) + list(self.data_path.glob("*.hdf5"))
            
            if not h5_files:
                raise ValueError(f"在目录 {self.data_path} 中未找到HDF5文件(.h5或.hdf5)")
                
            logger.info(f"找到 {len(h5_files)} 个HDF5文件")
            for file_path in h5_files:
                self._process_single_file(file_path)
        else:
            raise ValueError(f"无效的数据路径: {self.data_path}")
            
        # 限制样本数量
        if self.max_samples and len(self.sample_indices) > self.max_samples:
            self.sample_indices = self.sample_indices[:self.max_samples]
            
        logger.info(f"加载了 {len(self.sample_indices)} 个样本")
        
    def _process_single_file(self, file_path):
        """处理单个HDF5文件"""
        try:
            with h5py.File(str(file_path), 'r') as f:
                # 获取配置信息
                config = f.attrs.get('config', None)
                
                # 遍历所有数据集
                for key in f.keys():
                    if isinstance(f[key], h5py.Group):
                        # 如果是组（如seed_0000），则处理其中的数据
                        self._process_group(f[key], file_path, key, config)
                    elif isinstance(f[key], h5py.Dataset):
                        # 如果直接是数据集
                        self._process_dataset(f[key], file_path, key, config)
                        
        except Exception as e:
            logger.warning(f"处理文件 {file_path} 时出错: {e}")
            
    def _process_group(self, group, file_path, group_key, config):
        """处理HDF5组"""
        # 查找数据集
        data_key = None
        for key in group.keys():
            if key == 'data' or 'data' in key.lower():
                data_key = key
                break
                
        if data_key and isinstance(group[data_key], h5py.Dataset):
            dataset = group[data_key]
            # 获取数据形状
            shape = dataset.shape
            
            # 假设第一个维度是时间步
            if len(shape) >= 2:
                for t_idx in range(shape[0]):
                    self.sample_indices.append({
                        'file_path': file_path,
                        'group_key': group_key,
                        'data_key': data_key,
                        'time_idx': t_idx,
                        'shape': shape
                    })
                    self.configs.append(config)
                    
    def _process_dataset(self, dataset, file_path, data_key, config):
        """处理HDF5数据集"""
        shape = dataset.shape
        
        # 假设第一个维度是时间步或样本数
        if len(shape) >= 2:
            for idx in range(shape[0]):
                self.sample_indices.append({
                    'file_path': file_path,
                    'group_key': None,
                    'data_key': data_key,
                    'time_idx': idx,
                    'shape': shape
                })
                self.configs.append(config)
                
    def __len__(self):
        return len(self.sample_indices)
        
    def __getitem__(self, idx):
        sample_info = self.sample_indices[idx]
        config = self.configs[idx]
        
        # 加载数据
        with h5py.File(str(sample_info['file_path']), 'r') as f:
            if sample_info['group_key']:
                # 从组中加载数据
                group = f[sample_info['group_key']]
                data = np.array(group[sample_info['data_key']][sample_info['time_idx']])
                
                # 尝试加载网格信息
                grid_x = None
                grid_t = None
                if 'grid' in group:
                    if 'x' in group['grid']:
                        grid_x = np.array(group['grid']['x'])
                    if 't' in group['grid']:
                        grid_t = np.array(group['grid']['t'])
                        
            else:
                # 直接从数据集加载
                data = np.array(f[sample_info['data_key']][sample_info['time_idx']])
                grid_x = None
                grid_t = None
                
        # 转换为torch张量
        data = torch.from_numpy(data).float()
        
        # 展平数据以匹配现有模型输入格式
        if len(data.shape) > 1:
            data_flat = data.view(-1)
        else:
            data_flat = data
            
        # 创建输入和目标（这里简化处理，实际可能需要根据具体任务调整）
        # 假设输入是当前时间步，目标是下一时间步（或同一时间步的不同通道）
        input_data = data_flat
        target_data = data_flat  # 这里需要根据具体任务调整
        
        # 时间步信息
        time_step = sample_info['time_idx']
        
        # 应用变换
        if self.transform:
            input_data = self.transform(input_data)
            target_data = self.transform(target_data)
            
        return input_data, target_data, time_step
        
    def get_sample_info(self, idx):
        """获取样本信息"""
        return self.sample_indices[idx], self.configs[idx]
        
    def get_data_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.sample_indices),
            'unique_files': len(set(info['file_path'] for info in self.sample_indices)),
            'data_shapes': list(set(str(info['shape']) for info in self.sample_indices))
        }
        return stats