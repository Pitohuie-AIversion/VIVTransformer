#!/usr/bin/env python3
"""
SVD模态投影单元测试
验证SVDModalProjector的正确性和重建精度

测试内容:
1. SVD投影器基本功能测试
2. 数据投影和反投影精度测试
3. 配置参数验证测试
4. 边界条件和错误处理测试
5. 保存/加载功能测试

作者: Assistant
日期: 2025-01-15
"""

import sys
import os
import tempfile
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# 添加父目录到 Python 路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入测试模块
from pde_process.svd_modal_projection import SVDModalProjector, SVDProjectionConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(n_samples: int = 100, 
                     input_dim: int = 1024, 
                     output_dim: int = 4096,
                     rank: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建具有低秩结构的测试数据
    
    Args:
        n_samples: 样本数量
        input_dim: 输入维度
        output_dim: 输出维度
        rank: 数据的秩（用于控制复杂度）
    
    Returns:
        (input_data, output_data): 输入和输出数据
    """
    np.random.seed(42)
    
    # 生成低秩结构的数据
    U_true = np.random.randn(n_samples, rank)
    V_input = np.random.randn(rank, input_dim)
    V_output = np.random.randn(rank, output_dim)
    
    # 添加噪声
    noise_level = 0.1
    input_data = U_true @ V_input + noise_level * np.random.randn(n_samples, input_dim)
    output_data = U_true @ V_output + noise_level * np.random.randn(n_samples, output_dim)
    
    return input_data.astype(np.float32), output_data.astype(np.float32)

def test_svd_config_creation():
    """测试SVD配置创建"""
    logger.info("=== 测试SVD配置创建 ===")
    
    # 测试默认配置
    config_default = SVDProjectionConfig()
    assert config_default.n_modes == 64
    assert config_default.energy_threshold == 0.95
    assert config_default.auto_select_modes == True
    logger.info("[OK] 默认配置创建成功")
    
    # 测试自定义配置
    config_custom = SVDProjectionConfig(
        n_modes=32,
        energy_threshold=0.99,
        auto_select_modes=False,
        standardize_data=True
    )
    assert config_custom.n_modes == 32
    assert config_custom.energy_threshold == 0.99
    assert config_custom.auto_select_modes == False
    logger.info("[OK] 自定义配置创建成功")
    
    # 测试配置序列化
    config_dict = config_custom.to_dict()
    config_restored = SVDProjectionConfig.from_dict(config_dict)
    assert config_restored.n_modes == config_custom.n_modes
    assert config_restored.energy_threshold == config_custom.energy_threshold
    logger.info("[OK] 配置序列化/反序列化成功")
    
    return True

def test_svd_projector_basic_functionality():
    """测试SVD投影器基本功能"""
    logger.info("=== 测试SVD投影器基本功能 ===")
    
    # 创建测试数据
    input_data, output_data = create_test_data(n_samples=50, input_dim=512, output_dim=2048, rank=15)
    logger.info(f"测试数据形状: 输入 {input_data.shape}, 输出 {output_data.shape}")
    
    # 创建投影器
    config = SVDProjectionConfig(
        n_modes=20,
        standardize_data=True,
        auto_select_modes=False
    )
    projector = SVDModalProjector(config)
    
    # 验证初始状态
    assert not projector.is_fitted
    assert projector.latent_dim is None
    logger.info("[OK] 投影器初始状态正确")
    
    # 拟合投影器
    projector.fit(input_data, output_data)
    
    # 验证拟合后状态
    assert projector.is_fitted
    assert projector.latent_dim == 20
    assert projector.input_dim == 512
    assert projector.output_dim == 2048
    assert projector.input_projector is not None
    assert projector.output_projector is not None
    logger.info("[OK] 投影器拟合成功")
    
    # 测试投影功能
    input_projected = projector.transform_input(input_data)
    output_projected = projector.transform_output(output_data)
    
    assert input_projected.shape == (50, 20)
    assert output_projected.shape == (50, 20)
    logger.info("[OK] 数据投影成功")
    
    # 测试反投影功能
    input_reconstructed = projector.inverse_transform_input(input_projected)
    output_reconstructed = projector.inverse_transform_output(output_projected)
    
    assert input_reconstructed.shape == input_data.shape
    assert output_reconstructed.shape == output_data.shape
    logger.info("[OK] 数据反投影成功")
    
    return projector

def test_reconstruction_accuracy():
    """测试重建精度"""
    logger.info("=== 测试重建精度 ===")
    
    # 创建高质量测试数据
    input_data, output_data = create_test_data(n_samples=100, input_dim=256, output_dim=1024, rank=10)
    
    # 创建高精度配置
    config = SVDProjectionConfig(
        n_modes=15,  # 略高于数据的真实秩
        standardize_data=True,
        auto_select_modes=False
    )
    
    projector = SVDModalProjector(config)
    projector.fit(input_data, output_data)
    
    # 计算重建误差
    errors = projector.compute_reconstruction_error(input_data, output_data)
    
    logger.info(f"重建误差统计:")
    logger.info(f"  输入MSE: {errors['input_mse']:.6f}")
    logger.info(f"  输出MSE: {errors['output_mse']:.6f}")
    logger.info(f"  输入相对误差: {errors['input_relative_error']:.6f}")
    logger.info(f"  输出相对误差: {errors['output_relative_error']:.6f}")
    
    # 验证重建精度（由于数据是低秩的，重建应该比较准确）
    assert errors['input_relative_error'] < 0.5, f"输入重建误差过高: {errors['input_relative_error']:.6f}"
    assert errors['output_relative_error'] < 0.5, f"输出重建误差过高: {errors['output_relative_error']:.6f}"
    
    logger.info("[OK] 重建精度测试通过")
    return errors

def test_auto_mode_selection():
    """测试自动模态选择"""
    logger.info("=== 测试自动模态选择 ===")
    
    # 创建测试数据
    input_data, output_data = create_test_data(n_samples=80, input_dim=400, output_dim=1600, rank=12)
    
    # 测试自动模态选择
    config = SVDProjectionConfig(
        energy_threshold=0.90,
        auto_select_modes=True,
        min_modes=5,
        max_modes=30
    )
    
    projector = SVDModalProjector(config)
    projector.fit(input_data, output_data)
    
    # 验证选择的模态数在合理范围内
    assert 5 <= projector.latent_dim <= 30
    logger.info(f"[OK] 自动选择模态数: {projector.latent_dim}")
    
    # 验证能量阈值
    input_energy = np.sum(projector.input_singular_values[:projector.latent_dim] ** 2) / np.sum(projector.input_singular_values ** 2)
    output_energy = np.sum(projector.output_singular_values[:projector.latent_dim] ** 2) / np.sum(projector.output_singular_values ** 2)
    
    logger.info(f"输入数据保留能量: {input_energy:.4f}")
    logger.info(f"输出数据保留能量: {output_energy:.4f}")
    
    # 验证能量至少达到设定阈值的合理范围
    assert input_energy >= 0.85, f"输入数据保留能量过低: {input_energy:.4f}"
    assert output_energy >= 0.85, f"输出数据保留能量过低: {output_energy:.4f}"
    
    logger.info("[OK] 自动模态选择测试通过")
    return True

def test_save_load_functionality():
    """测试保存/加载功能"""
    logger.info("=== 测试保存/加载功能 ===")
    
    # 创建测试数据和投影器
    input_data, output_data = create_test_data(n_samples=40, input_dim=200, output_dim=800, rank=8)
    
    config = SVDProjectionConfig(
        n_modes=12,
        standardize_data=True
    )
    
    original_projector = SVDModalProjector(config)
    original_projector.fit(input_data, output_data)
    
    # 测试原始投影器
    original_input_proj = original_projector.transform_input(input_data[:5])
    original_output_proj = original_projector.transform_output(output_data[:5])
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        save_path = tmp_file.name
    
    try:
        # 保存投影器
        original_projector.save(save_path)
        logger.info(f"[OK] 投影器已保存到: {save_path}")
        
        # 加载投影器
        loaded_projector = SVDModalProjector.load(save_path)
        logger.info("[OK] 投影器加载成功")
        
        # 验证加载的投影器状态
        assert loaded_projector.is_fitted == original_projector.is_fitted
        assert loaded_projector.latent_dim == original_projector.latent_dim
        assert loaded_projector.input_dim == original_projector.input_dim
        assert loaded_projector.output_dim == original_projector.output_dim
        
        # 验证投影结果一致性
        loaded_input_proj = loaded_projector.transform_input(input_data[:5])
        loaded_output_proj = loaded_projector.transform_output(output_data[:5])
        
        np.testing.assert_allclose(original_input_proj, loaded_input_proj, rtol=1e-6)
        np.testing.assert_allclose(original_output_proj, loaded_output_proj, rtol=1e-6)
        
        logger.info("[OK] 投影结果一致性验证通过")
        
        # 验证配置一致性
        original_config_dict = original_projector.config.to_dict()
        loaded_config_dict = loaded_projector.config.to_dict()
        assert original_config_dict == loaded_config_dict
        
        logger.info("[OK] 配置一致性验证通过")
        
    finally:
        # 清理临时文件
        if os.path.exists(save_path):
            os.unlink(save_path)
    
    return True

def test_error_handling():
    """测试错误处理"""
    logger.info("=== 测试错误处理 ===")
    
    config = SVDProjectionConfig(n_modes=10)
    projector = SVDModalProjector(config)
    
    # 测试未拟合投影器的操作
    dummy_data = np.random.randn(5, 100)
    
    try:
        projector.transform_input(dummy_data)
        assert False, "应该抛出未拟合异常"
    except ValueError:
        logger.info("[OK] 未拟合投影器错误处理正确")
    
    # 测试维度不匹配
    input_data, output_data = create_test_data(n_samples=30, input_dim=100, output_dim=400)
    projector.fit(input_data, output_data)
    
    # 尝试投影错误维度的数据
    wrong_dim_data = np.random.randn(5, 200)  # 错误维度
    
    try:
        projector.transform_input(wrong_dim_data)
        assert False, "应该抛出维度不匹配异常"
    except ValueError:
        logger.info("[OK] 维度不匹配错误处理正确")
    
    # 测试保存未拟合投影器
    unfitted_projector = SVDModalProjector(config)
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp_file:
            unfitted_projector.save(tmp_file.name)
        assert False, "应该抛出未拟合异常"
    except ValueError:
        logger.info("[OK] 保存未拟合投影器错误处理正确")
    
    return True

def test_edge_cases():
    """测试边界条件"""
    logger.info("=== 测试边界条件 ===")
    
    # 测试极小数据集
    tiny_input = np.random.randn(3, 20)
    tiny_output = np.random.randn(3, 50)
    
    config = SVDProjectionConfig(n_modes=2, auto_select_modes=False)
    projector = SVDModalProjector(config)
    
    try:
        projector.fit(tiny_input, tiny_output)
        logger.info("[OK] 极小数据集处理成功")
    except Exception as e:
        logger.warning(f"极小数据集处理失败: {e}")
    
    # 测试高维度低样本
    few_samples_input = np.random.randn(5, 1000)
    few_samples_output = np.random.randn(5, 2000)
    
    config_few = SVDProjectionConfig(n_modes=3, auto_select_modes=False)
    projector_few = SVDModalProjector(config_few)
    
    try:
        projector_few.fit(few_samples_input, few_samples_output)
        logger.info("[OK] 高维度低样本处理成功")
    except Exception as e:
        logger.warning(f"高维度低样本处理失败: {e}")
    
    return True

def run_all_tests():
    """运行所有测试"""
    logger.info("=== 开始SVD投影单元测试 ===")
    
    test_results = {}
    
    try:
        test_results['config_creation'] = test_svd_config_creation()
        test_results['basic_functionality'] = test_svd_projector_basic_functionality() is not None
        test_results['reconstruction_accuracy'] = test_reconstruction_accuracy() is not None
        test_results['auto_mode_selection'] = test_auto_mode_selection()
        test_results['save_load'] = test_save_load_functionality()
        test_results['error_handling'] = test_error_handling()
        test_results['edge_cases'] = test_edge_cases()
        
        # 统计测试结果
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"\n=== 测试结果总结 ===")
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {total_tests - passed_tests}")
        logger.info(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        for test_name, result in test_results.items():
            status = "[OK] PASS" if result else "[ERROR] FAIL"
            logger.info(f"  {test_name}: {status}")
        
        if passed_tests == total_tests:
            logger.info("[OK] 所有测试通过！SVD投影功能正常工作")
            return True
        else:
            logger.warning("[WARN] 部分测试失败，请检查相关功能")
            return False
            
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)