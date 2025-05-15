from torch.utils.data import DataLoader, random_split
from .dataset import PressureDataset
import torch

def collate_fn(batch):
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if batch:
        # Dataset 返回 (in_press, target, x_time_steps, mask, weight_maps)
        inputs, targets, x_time_steps, masks, weight_maps = \
            torch.utils.data.dataloader.default_collate(batch)
        return inputs, targets, x_time_steps, masks, weight_maps
    return None

def get_loaders(data_path, batch_size, num_workers=0, pin_memory=False):
    """
    创建并返回训练/验证/测试 DataLoader。
    :param data_path: 数据文件路径，传给 PressureDataset
    :param batch_size: 每个 batch 的样本数
    :param num_workers: DataLoader 的工作进程数
    :param pin_memory: 是否将数据锁页到内存
    :return: train_loader, valid_loader, test_loader, input_dim, output_dim
    """
    # 实例化 Dataset
    dataset = PressureDataset(data_path)

    # 从 dataset 中获取维度信息
    input_dim  = dataset.input_dim
    output_dim = dataset.output_dim

    # 划分数据集
    total      = len(dataset)
    train_size = int(0.7 * total)
    valid_size = int(0.2 * total)
    test_size  = total - train_size - valid_size
    train_set, valid_set, test_set = random_split(
        dataset, [train_size, valid_size, test_size]
    )

    # 创建 DataLoader 并传入自定义 collate_fn
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader, input_dim, output_dim
