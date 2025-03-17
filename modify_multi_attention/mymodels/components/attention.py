import torch
import torch.nn as nn


class RelativePositionSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=500):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.max_len = max_len

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 初始化相对位置嵌入
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(num_heads, 2 * max_len - 1)
        )

    def forward(self, query, key, value):
        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()

        q = self.query(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(value).view(batch_size, key_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 计算正确的相对位置索引
        position_ids_q = torch.arange(query_len, dtype=torch.long, device=query.device)
        position_ids_k = torch.arange(key_len, dtype=torch.long, device=key.device)
        relative_position = position_ids_q.unsqueeze(-1) - position_ids_k.unsqueeze(0) + self.max_len - 1  # 偏移以确保索引>=0

        relative_position_embeddings = self.relative_position_embeddings[:, relative_position]
        scores += relative_position_embeddings.unsqueeze(0)

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        return output




class SparseSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.block_size = block_size

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()

        q = self.query(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(value).view(batch_size, key_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 创建稀疏掩码
        mask = torch.ones_like(scores, dtype=torch.bool)  # 使用 dtype=torch.bool 以适配 masked_fill
        for i in range(query_len):
            start = (i // self.block_size) * self.block_size
            end = min(start + self.block_size, key_len)  # 确保 end 不超过 key_len
            mask[:, :, i, start:end] = False  # 仅保留 block_size 内的注意力

        scores = scores.masked_fill(mask, float('-inf'))  # 只允许 block_size 范围内的注意力

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        return output





class LSHSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_hash_functions=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_hash_functions = num_hash_functions

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def hash_function(self, x):
        """
        计算 LSH 哈希编码，保证 batch, heads, seq_len 形状匹配
        """
        batch_size, num_heads, seq_len, _ = x.shape  # 获取 batch, heads, seq_len
        return torch.randint(0, 2, (batch_size, num_heads, seq_len), device=x.device)

    def forward(self, query, key, value):
        batch_size, query_len, _ = query.size()
        _, key_len, _ = key.size()

        q = self.query(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, query_len, d_k]
        k = self.key(key).view(batch_size, key_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch, heads, key_len, d_k]
        v = self.value(value).view(batch_size, key_len, self.num_heads, self.d_v).transpose(1, 2)  # [batch, heads, key_len, d_v]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [batch, heads, query_len, key_len]

        # 计算哈希码，并确保它们的形状一致
        hash_codes_q = self.hash_function(q)  # [batch, heads, query_len]
        hash_codes_k = self.hash_function(k)  # [batch, heads, key_len]

        # 生成哈希掩码，确保掩码维度和 scores 匹配
        mask = (hash_codes_q.unsqueeze(-1) != hash_codes_k.unsqueeze(-2))  # [batch, heads, query_len, key_len]
        mask = mask.float() * -1e9  # 只允许哈希相等的元素注意

        # 应用哈希掩码
        scores = scores + mask

        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        return output
