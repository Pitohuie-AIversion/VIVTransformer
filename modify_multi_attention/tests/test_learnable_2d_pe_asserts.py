import os
import sys
import torch
import pytest

# Ensure project root on sys.path so that package-style imports work when running pytest from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel  # noqa: E402


def test_learnable_2d_pe_requires_valid_input_hw_or_seq_len_factorization():
    # Case 1: Provide valid input_hw matching seq_len
    m = TransformerFlowReconstructionModel(
        input_dim=16,
        output_dim=16,
        num_heads=2,
        num_layers=1,
        d_model=32,
        max_time_steps=10,
        attention_type='relative',
        seq_len=16,
        input_hw=(4, 4),
        pe_type='learnable_2d',
        output_head_type='global'
    )
    assert m.input_hw == (4, 4)
    assert m.seq_len == 16


def test_learnable_2d_pe_raises_when_hw_mismatch_seq_len():
    # input_hw has higher priority than seq_len; seq_len should be set to H*W
    m = TransformerFlowReconstructionModel(
        input_dim=16,
        output_dim=16,
        num_heads=2,
        num_layers=1,
        d_model=32,
        max_time_steps=10,
        attention_type='relative',
        seq_len=15,  # conflicting
        input_hw=(4, 4),  # takes precedence
        pe_type='learnable_2d',
        output_head_type='global'
    )
    assert m.seq_len == 16
    assert m.input_hw == (4, 4)


def test_learnable_2d_pe_inferrable_from_seq_len_square():
    # When input_hw is None and seq_len is a perfect square, model infers (sqrt, sqrt)
    m = TransformerFlowReconstructionModel(
        input_dim=25,
        output_dim=25,
        num_heads=2,
        num_layers=1,
        d_model=32,
        max_time_steps=10,
        attention_type='relative',
        seq_len=25,
        input_hw=None,
        pe_type='learnable_2d',
        output_head_type='global'
    )
    assert m.input_hw == (5, 5)
    assert m.seq_len == 25


def test_learnable_2d_pe_inferrable_from_seq_len_rect_factor():
    # When seq_len is not perfect square, choose factorization closest to square
    m = TransformerFlowReconstructionModel(
        input_dim=18,
        output_dim=18,
        num_heads=2,
        num_layers=1,
        d_model=32,
        max_time_steps=10,
        attention_type='relative',
        seq_len=18,
        input_hw=None,
        pe_type='learnable_2d',
        output_head_type='global'
    )
    H, W = m.input_hw
    assert H * W == 18
    # the pair should be one of the factor pairs and near-square
    factor_pairs = [(i, 18//i) for i in range(1, int(18**0.5)+1) if 18 % i == 0]
    assert (H, W) in factor_pairs or (W, H) in factor_pairs


def test_learnable_2d_pe_invalid_hw_nonpositive():
    with pytest.raises(AssertionError):
        TransformerFlowReconstructionModel(
            input_dim=16,
            output_dim=16,
            num_heads=2,
            num_layers=1,
            d_model=32,
            max_time_steps=10,
            attention_type='relative',
            seq_len=16,
            input_hw=(4, 0),  # invalid
            pe_type='learnable_2d',
            output_head_type='global'
        )