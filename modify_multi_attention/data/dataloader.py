# from torch.utils.data import DataLoader, random_split  # 导入 random_split
# from .dataset import PressureDataset
# import torch
#
# def collate_fn(batch):
#     """
#     处理每个批次数据，返回包含掩码的数据。
#     :param batch: 数据批次
#     :return: 处理后的批次数据
#     """
#     batch = [b for b in batch if b is not None]
#
#     if batch:
#         inputs, targets, time_steps, mask = torch.utils.data.dataloader.default_collate(batch)
#         return inputs, targets, time_steps, mask  # 返回 4 个元素
#     return None
#
# def get_loaders(data_path, batch_size):
#     """
#     获取训练、验证和测试集的加载器
#     :param data_path: 数据路径
#     :param batch_size: 批次大小
#     :return: 训练、验证、测试集的 DataLoader
#     """
#     dataset = PressureDataset(data_path)
#
#     # 动态获取输入输出维度
#     input_dim = dataset.input_dim
#     output_dim = dataset.output_dim
#
#     train_size = int(0.7 * len(dataset))
#     valid_size = int(0.2 * len(dataset))
#     test_size = len(dataset) - train_size - valid_size
#     train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])
#
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, valid_loader, test_loader, input_dim, output_dim
from torch.utils.data import DataLoader, random_split
from .dataset import PressureDataset
import torch

def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    if batch:
        inputs, targets, time_steps, mask, reynolds = torch.utils.data.dataloader.default_collate(batch)
        return inputs, targets, time_steps, mask, reynolds
    return None

def get_loaders(data_path, batch_size):
    dataset = PressureDataset(data_path)

    input_dim = dataset.input_dim
    output_dim = dataset.output_dim

    train_size = int(0.7 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader, input_dim, output_dim
