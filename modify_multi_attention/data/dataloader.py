import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from .dataset import PressureDataset
from .pdebench_dataset import PDEBenchDataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


class ToyDataset(Dataset):
    """
    一个用于冒烟测试的轻量级随机数据集。
    - 输入维度默认 400，与默认配置的 model.input_dim 一致
    - 输出维度默认 40000，与默认配置的 model.output_dim 一致
    - 样本数量默认 64，可通过 max_samples 控制
    - 时间步为简单的循环整数标量
    """
    def __init__(self, input_dim: int = 400, output_dim: int = 40000, length: int = 64, time_steps: int = 10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.length = length
        self.time_steps = max(1, int(time_steps))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim, dtype=torch.float32)
        y = torch.randn(self.output_dim, dtype=torch.float32)
        t = torch.tensor(idx % self.time_steps, dtype=torch.long)
        return x, y, t


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
        dataset_type: 数据集类型 ('auto', 'pressure', 'pdebench', 'toy')
        max_samples: 最大样本数量限制（对任意数据集生效；若为None则不限制）
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
    elif dataset_type == 'toy':
        toy_len = max_samples if (max_samples is not None and max_samples > 0) else 64
        dataset = ToyDataset(length=toy_len)
        logger.info(f"使用 ToyDataset 进行冒烟测试，长度={len(dataset)}，输入维度={dataset.input_dim}，输出维度={dataset.output_dim}")
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    # 打印数据集基本信息
    logger.info(f"="*60)
    logger.info(f"数据集加载信息 (Dataset Loading Information)")
    logger.info(f"="*60)
    logger.info(f"数据集类型 (Dataset Type): {dataset_type}")
    logger.info(f"原始数据集大小 (Original Dataset Size): {len(dataset)} 个样本")
    
    # 获取并打印数据维度信息
    try:
        if len(dataset) > 0:
            sample_data = dataset[0]
            if isinstance(sample_data, (tuple, list)) and len(sample_data) >= 2:
                input_data, target_data = sample_data[0], sample_data[1]
                input_shape = input_data.shape if hasattr(input_data, 'shape') else len(input_data)
                target_shape = target_data.shape if hasattr(target_data, 'shape') else len(target_data)
                logger.info(f"输入数据维度 (Input Data Shape): {input_shape}")
                logger.info(f"目标数据维度 (Target Data Shape): {target_shape}")
                
                # 计算数据大小
                input_size = input_data.numel() if hasattr(input_data, 'numel') else len(input_data)
                target_size = target_data.numel() if hasattr(target_data, 'numel') else len(target_data)
                logger.info(f"输入数据元素数量 (Input Data Elements): {input_size}")
                logger.info(f"目标数据元素数量 (Target Data Elements): {target_size}")
            else:
                logger.info(f"样本数据格式: {type(sample_data)}")
    except Exception as e:
        logger.warning(f"获取数据维度信息时出错: {e}")
    
    # 通用的 max_samples 支持：对任意数据集进行子集限制
    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
        logger.info(f"已应用 max_samples={max_samples}，限制后数据集大小: {len(dataset)} 个样本")
    
    # 如果是PDEBench数据集，打印详细统计信息
    if isinstance(getattr(dataset, 'dataset', dataset), PDEBenchDataset):
        # 兼容 Subset 包裹的情况
        base_ds = dataset.dataset if isinstance(dataset, Subset) else dataset
        stats = base_ds.get_data_statistics()
        logger.info(f"PDEBench数据集详细统计:")
        logger.info(f"  - 总样本数: {stats['total_samples']}")
        logger.info(f"  - 文件数量: {stats['unique_files']}")
        logger.info(f"  - 数据形状: {stats['data_shapes']}")
        
        # 验证是否为10000个样本的真实数据
        if stats['total_samples'] == 10000:
            logger.info(f"✓ 确认: 这是包含10000个样本的真实数据集!")
        elif stats['total_samples'] > 10000:
            logger.info(f"ℹ️ 注意: 数据集包含{stats['total_samples']}个样本，超过预期的10000个")
        else:
            logger.info(f"⚠️ 注意: 数据集只包含{stats['total_samples']}个样本，少于预期的10000个")
    
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
    
    # 打印数据分割信息
    logger.info(f"="*60)
    logger.info(f"数据分割信息 (Data Split Information)")
    logger.info(f"="*60)
    logger.info(f"训练集 (Training Set): {len(train_dataset)} 个样本 ({len(train_loader)} 个批次)")
    logger.info(f"验证集 (Validation Set): {len(valid_dataset)} 个样本 ({len(valid_loader)} 个批次)")
    logger.info(f"测试集 (Test Set): {len(test_dataset)} 个样本 ({len(test_loader)} 个批次)")
    logger.info(f"批次大小 (Batch Size): {batch_size}")
    logger.info(f"数据分割比例 - 训练:验证:测试 = {train_ratio}:{valid_ratio}:{test_ratio}")
    logger.info(f"="*60)
    
    return train_loader, valid_loader, test_loader
