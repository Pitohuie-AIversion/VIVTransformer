import torch
from torch.utils.data import DataLoader, random_split
from .dataset import PressureDataset
from .pdebench_dataset import PDEBenchDataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


def get_loaders(data_path, batch_size, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, 
                dataset_type='auto', max_samples=None):
    """
    获取数据加载器
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        dataset_type: 数据集类型 ('auto', 'pressure', 'pdebench')
        max_samples: 最大样本数量限制
    """
    
    # 自动检测数据集类型
    if dataset_type == 'auto':
        data_path_obj = Path(data_path)
        if data_path_obj.suffix in ['.h5', '.hdf5'] or \
           (data_path_obj.is_dir() and (list(data_path_obj.glob('*.h5')) or list(data_path_obj.glob('*.hdf5')))):
            dataset_type = 'pdebench'
            logger.info("检测到HDF5格式，使用PDEBench数据集")
        elif data_path_obj.suffix == '.pt':
            dataset_type = 'pressure'
            logger.info("检测到PyTorch格式，使用Pressure数据集")
        else:
            raise ValueError(f"无法识别数据格式: {data_path}")
    
    # 创建数据集
    if dataset_type == 'pressure':
        dataset = PressureDataset(data_path)
    elif dataset_type == 'pdebench':
        dataset = PDEBenchDataset(data_path, max_samples=max_samples)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 如果是PDEBench数据集，打印统计信息
    if isinstance(dataset, PDEBenchDataset):
        stats = dataset.get_data_statistics()
        logger.info(f"数据集统计: {stats}")
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(valid_dataset)}, 测试集: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader
