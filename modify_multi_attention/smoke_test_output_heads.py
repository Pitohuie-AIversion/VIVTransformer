import sys
from pathlib import Path
import torch

# Ensure project root and module paths
root = Path(r"X:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1")
sys.path.insert(0, str(root / 'modify_multi_attention'))
sys.path.insert(0, str(root))

from mymodels.transformer import TransformerFlowReconstructionModel

def main():
    B = 2
    H = W = 32
    in_dim = H * W
    seq_len = H

    # Global head
    out_dim_global = 1
    m1 = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=out_dim_global,
        num_heads=4,
        num_layers=1,
        d_model=64,
        max_time_steps=50,
        attention_type='sge',
        seq_len=seq_len,
        input_hw=(H, W),
        pe_type='learnable_1d',
        output_head_type='global'
    )

    x = torch.randn(B, in_dim)
    t = torch.randint(0, 50, (B,))
    y1 = m1(x, t)
    print('Global OK:', y1.shape)

    # Per-token head
    per_ch = 3
    out_dim_token = (H * W) * per_ch
    m2 = TransformerFlowReconstructionModel(
        input_dim=in_dim,
        output_dim=out_dim_token,
        num_heads=4,
        num_layers=1,
        d_model=64,
        max_time_steps=50,
        attention_type='sge',
        seq_len=seq_len,
        input_hw=(H, W),
        pe_type='learnable_2d',
        output_head_type='per_token',
        out_channels_per_token=per_ch
    )
    y2 = m2(x, t)
    print('Per-token OK:', y2.shape)

if __name__ == '__main__':
    main()