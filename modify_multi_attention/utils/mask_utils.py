import torch

def generate_box_mask(batch_size, height, width, boxes):
    """
    生成掩码张量，支持多个矩形区域框选，支持展平后的数据。
    :param batch_size: 批次大小
    :param height: 输出图像的高度
    :param width: 输出图像的宽度
    :param boxes: 形如 [[x_start, x_end, y_start, y_end], ...] 的列表
    :return: 掩码张量，大小为 (batch_size, height * width)
    """
    mask = torch.zeros((batch_size, height * width), dtype=torch.float32)  # 初始化全零掩码
    for x_start, x_end, y_start, y_end in boxes:
        # 将二维坐标映射到展平后的索引
        for batch_idx in range(batch_size):
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    # 计算展平后的一维索引
                    index = y * width + x
                    mask[batch_idx, index] = 1.0  # 设置掩码区域为1
    return mask
