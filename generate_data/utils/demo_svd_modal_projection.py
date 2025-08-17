#!/usr/bin/env python3
"""
SVD模态分解综合演示脚本
展示SVD模态投影器的完整使用流程：数据加载、训练、投影、重建和可视化

Usage:
    python demo_svd_modal_projection.py [--data-path path/to/data.h5] [--n-modes 64] [--max-samples 200]
"""

import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
import warnings
from pathlib import Path
import sys
import os
import h5py
import json
from typing import Dict, List, Tuple, Optional

# 添加父目录到 Python 路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入SVD投影模块
from pde_process.svd_modal_projection import SVDModalProjector, SVDProjectionConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置中文字体
def configure_chinese_font():
    """配置中文字体支持（更健壮）：
    - 尝试注册系统常见中文字体文件（Windows 常见路径）
    - 在候选列表中选择第一个可用的中文字体
    - 兼容 seaborn 的样式覆盖（函数可重复调用）
    - 忽略缺字形的噪声警告
    """
    try:
        # 过滤字体相关警告以减少噪声
        warnings.filterwarnings("ignore", message=r".*Glyph.*missing from font.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

        # 候选中文字体名称（按优先级）
        candidate_fonts = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "PingFang SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]

        # 在 Windows 上尝试显式注册常见字体文件，保证能被 Matplotlib 检测到
        possible_font_files = [
            r"C:\\Windows\\Fonts\\msyh.ttc",
            r"C:\\Windows\\Fonts\\msyh.ttf",
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc",
            r"C:\\Windows\\Fonts\\SourceHanSansSC-Regular.otf",
        ]
        for fp in possible_font_files:
            if os.path.exists(fp):
                try:
                    fm.fontManager.addfont(fp)
                except Exception:
                    pass

        # 构建可用字体名称集合并选择一个可用中文字体
        try:
            available_names = {f.name for f in fm.fontManager.ttflist}
        except Exception:
            available_names = set()

        selected = None
        for name in candidate_fonts:
            if name in available_names:
                selected = name
                break
        if selected is None:
            # 若仍未找到中文字体，退回到 DejaVu Sans（不含全量中文，仅避免崩溃）
            selected = "DejaVu Sans"

        # 应用 rcParams 设置（可重复调用，抵消 seaborn 的覆盖）
        matplotlib.rcParams['font.family'] = 'sans-serif'
        # 统一使用全局 sitecustomize.py 的中文字体与负号设置
# matplotlib.rcParams['font.sans-serif'] = [selected] + [f for f in candidate_fonts if f != selected]
# matplotlib.rcParams['axes.unicode_minus'] = False

        try:
            logger.info(f"Matplotlib 中文字体设置: preferred='{selected}'")
        except Exception:
            pass
    except Exception as e:
        try:
            logger.warning(f"字体配置失败: {e}")
        except Exception:
            pass

configure_chinese_font()
# seaborn 的样式可能会部分覆盖字体相关设置，先设置样式，再次调用字体配置进行加固
sns.set_style("whitegrid")
configure_chinese_font()

def load_hdf5_data(data_path: Path, max_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载HDF5数据并预处理
    
    Args:
        data_path: HDF5文件路径
        max_samples: 最大样本数
        
    Returns:
        输入数据和输出数据
    """
    logger.info(f"加载HDF5数据: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # 获取数据集信息
        logger.info(f"HDF5文件内容: {list(f.keys())}")
        
        # 加载数据（假设数据在'data'键下）
        if 'data' in f:
            data = np.array(f['data'])
            logger.info(f"原始数据形状: {data.shape}")
            
            # 限制样本数量
            if data.shape[0] > max_samples:
                indices = np.random.choice(data.shape[0], max_samples, replace=False)
                data = data[indices]
                logger.info(f"随机选择 {max_samples} 个样本")
            
            # 根据数据维度划分输入输出
            if len(data.shape) == 4:  # (N, C, H, W)
                n_samples, n_channels, height, width = data.shape
                
                # 将前面的通道作为输入，后面的作为输出
                if n_channels >= 2:
                    input_channels = n_channels // 2
                    input_data = data[:, :input_channels].reshape(n_samples, -1)
                    output_data = data[:, input_channels:].reshape(n_samples, -1)
                else:
                    # 如果只有一个通道，创建简单的输入输出关系
                    input_data = data[:, 0].reshape(n_samples, -1)
                    # 输出为输入的下采样+变换
                    output_data = input_data[:, ::4]  # 简单下采样作为输出
                    
            elif len(data.shape) == 3:  # (N, H, W)
                n_samples, height, width = data.shape
                input_data = data.reshape(n_samples, -1)
                # 创建人工输出（比如上采样或变换）
                output_data = np.repeat(input_data, 2, axis=1)  # 简单重复扩展
                
            else:
                raise ValueError(f"不支持的数据形状: {data.shape}")
                
        else:
            raise ValueError("HDF5文件中没有找到'data'键")
    
    logger.info(f"预处理后数据形状:")
    logger.info(f"  输入: {input_data.shape}")
    logger.info(f"  输出: {output_data.shape}")
    
    return input_data, output_data

def create_synthetic_data(n_samples: int = 100, input_dim: int = 1024, output_dim: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建合成数据用于演示
    
    Args:
        n_samples: 样本数量
        input_dim: 输入维度
        output_dim: 输出维度
        
    Returns:
        输入数据和输出数据
    """
    logger.info(f"创建合成数据: {n_samples} 样本, 输入维度 {input_dim}, 输出维度 {output_dim}")
    
    np.random.seed(42)
    
    # 创建低秩结构的数据
    latent_dim = 10
    latent_factors = np.random.randn(n_samples, latent_dim)
    
    # 输入数据：潜在因子的线性组合
    input_basis = np.random.randn(latent_dim, input_dim)
    input_data = latent_factors @ input_basis
    input_data += 0.1 * np.random.randn(n_samples, input_dim)  # 添加噪声
    
    # 输出数据：潜在因子的非线性变换
    output_basis = np.random.randn(latent_dim, output_dim)
    output_data = np.tanh(latent_factors * 1.5) @ output_basis
    output_data += 0.05 * np.random.randn(n_samples, output_dim)  # 添加噪声
    
    return input_data, output_data

def demonstrate_svd_workflow(input_data, output_data, n_modes, output_dir):
    """演示完整的SVD工作流程"""
    
    logger.info("开始SVD模态分解演示...")
    logger.info(f"输入数据形状: {input_data.shape}")
    logger.info(f"输出数据形状: {output_data.shape}")
    logger.info(f"目标模态数: {n_modes}")
    
    # 1. 配置SVD投影器
    logger.info("1. 配置SVD投影器...")
    config = SVDProjectionConfig(
        n_modes=n_modes,
        energy_threshold=0.99
    )
    projector = SVDModalProjector(config)
    
    # 2. 训练投影器
    logger.info("2. 训练SVD投影器...")
    projector.fit(input_data, output_data)
    
    # 3. 获取投影信息
    logger.info("3. 分析投影信息...")
    projection_info = projector.get_projection_info()
    logger.info(f"   - 输入维度: {projection_info['input_dim']}")
    logger.info(f"   - 输出维度: {projection_info['output_dim']}")
    logger.info(f"   - 潜在维度: {projection_info['latent_dim']}")
    # 去除不存在的能量保持键，避免KeyError
    # 可在可视化中展示能量分布
    
    # 4. 执行投影
    logger.info("4. 执行前向投影...")
    input_projected = projector.transform_input(input_data)
    output_projected = projector.transform_output(output_data)
    
    # 5. 执行重建
    logger.info("5. 执行反向重建...")
    input_reconstructed = projector.inverse_transform_input(input_projected)
    output_reconstructed = projector.inverse_transform_output(output_projected)
    
    # 6. 计算重建误差
    logger.info("6. 计算重建误差...")
    reconstruction_errors = {
        'input_mse': np.mean((input_data - input_reconstructed) ** 2),
        'output_mse': np.mean((output_data - output_reconstructed) ** 2),
        'input_relative_error': np.linalg.norm(input_data - input_reconstructed) / np.linalg.norm(input_data),
        'output_relative_error': np.linalg.norm(output_data - output_reconstructed) / np.linalg.norm(output_data)
    }
    
    logger.info(f"   - 输入MSE: {reconstruction_errors['input_mse']:.6f}")
    logger.info(f"   - 输出MSE: {reconstruction_errors['output_mse']:.6f}")
    logger.info(f"   - 输入相对误差: {reconstruction_errors['input_relative_error']:.6f}")
    logger.info(f"   - 输出相对误差: {reconstruction_errors['output_relative_error']:.6f}")
    
    # 7. 可视化结果
    logger.info("7. 生成可视化...")
    # 重新配置中文字体以防被其他库覆盖
    configure_chinese_font()
    
    try:
        viz_path = visualize_svd_results(
            projector, input_data, output_data,
            input_projected, output_projected,
            input_reconstructed, output_reconstructed,
            reconstruction_errors, output_dir
        )
        logger.info(f"可视化完成: {viz_path}")
    except Exception as e:
        logger.error(f"可视化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 8. 保存投影器
    logger.info("8. 保存投影器...")
    projector_path = output_dir / 'svd_projector.pkl'
    projector.save(projector_path)
    logger.info(f"投影器已保存到: {projector_path}")
    
    # 9. 生成报告
    logger.info("8. 生成分析报告...")
    try:
        generate_demo_report(projector, reconstruction_errors, output_dir)
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
    
    return projector, reconstruction_errors

def visualize_svd_results(projector, input_data, output_data, 
                         input_projected, output_projected,
                         input_reconstructed, output_reconstructed,
                         reconstruction_errors, output_dir):
    """可视化SVD分解结果"""
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"可视化开始，输出目录: {output_dir}")
    
    # 创建大图
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('SVD模态分解综合演示结果', fontsize=16, fontweight='bold')
    
    # 1. 奇异值衰减
    ax = axes[0, 0]
    n_show = min(50, len(projector.input_singular_values))
    ax.semilogy(range(1, n_show+1), projector.input_singular_values[:n_show], 
               'b-o', label='输入奇异值', markersize=4)
    ax.semilogy(range(1, n_show+1), projector.output_singular_values[:n_show], 
               'r-s', label='输出奇异值', markersize=4)
    ax.set_xlabel('模态索引')
    ax.set_ylabel('奇异值 (对数尺度)')
    ax.set_title('奇异值衰减曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 能量比例
    ax = axes[0, 1]
    input_energy = (projector.input_singular_values ** 2) / np.sum(projector.input_singular_values ** 2)
    output_energy = (projector.output_singular_values ** 2) / np.sum(projector.output_singular_values ** 2)
    n_show = min(20, len(input_energy))
    
    ax.bar(range(1, n_show+1), input_energy[:n_show], alpha=0.7, label='输入', color='blue')
    ax.bar(range(1, n_show+1), output_energy[:n_show], alpha=0.7, label='输出', color='red')
    ax.set_xlabel('模态索引')
    ax.set_ylabel('能量比例')
    ax.set_title('模态能量分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 累积能量
    ax = axes[0, 2]
    input_cumulative = np.cumsum(input_energy)
    output_cumulative = np.cumsum(output_energy)
    
    ax.plot(range(1, len(input_cumulative)+1), input_cumulative, 'b-', label='输入累积能量', linewidth=2)
    ax.plot(range(1, len(output_cumulative)+1), output_cumulative, 'r-', label='输出累积能量', linewidth=2)
    ax.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=0.95, color='k', linestyle=':', alpha=0.5, label='95%')
    ax.set_xlabel('模态数量')
    ax.set_ylabel('累积能量比例')
    ax.set_title('累积能量保持')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # 4. 重建误差
    ax = axes[0, 3]
    error_names = ['输入MSE', '输出MSE', '输入相对误差', '输出相对误差']
    error_values = [
        reconstruction_errors['input_mse'],
        reconstruction_errors['output_mse'],
        reconstruction_errors['input_relative_error'],
        reconstruction_errors['output_relative_error']
    ]
    bars = ax.bar(error_names, error_values, color=['blue', 'red', 'lightblue', 'lightcoral'])
    ax.set_ylabel('误差值')
    ax.set_title('重建误差统计')
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, error_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
               f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 5-8. 数据样本可视化（展示前4个样本的前100个特征）
    sample_indices = [0, 1, 2, 3]
    feature_range = slice(0, min(100, input_data.shape[1]))
    
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[1, i]
        
        # 原始输入 vs 重建输入
        ax.plot(input_data[sample_idx, feature_range], 'b-', label='原始输入', alpha=0.7)
        ax.plot(input_reconstructed[sample_idx, feature_range], 'r--', label='重建输入', alpha=0.7)
        ax.set_xlabel('特征索引')
        ax.set_ylabel('特征值')
        ax.set_title(f'样本 {sample_idx+1} 输入重建')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 9-12. 输出数据重建可视化
    output_feature_range = slice(0, min(100, output_data.shape[1]))
    
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[2, i]
        
        # 原始输出 vs 重建输出
        ax.plot(output_data[sample_idx, output_feature_range], 'b-', label='原始输出', alpha=0.7)
        ax.plot(output_reconstructed[sample_idx, output_feature_range], 'r--', label='重建输出', alpha=0.7)
        ax.set_xlabel('特征索引')
        ax.set_ylabel('特征值')
        ax.set_title(f'样本 {sample_idx+1} 输出重建')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像，并添加更详细的日志
    output_path = output_dir / 'svd_demo_results.png'
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"图片已保存到: {output_path}")
        logger.info(f"文件大小: {output_path.stat().st_size if output_path.exists() else '保存失败'} 字节")
    except Exception as e:
        logger.error(f"保存图片失败: {e}")
    finally:
        plt.close()
    
    return output_path

def generate_demo_report(projector, reconstruction_errors, output_dir):
    """生成演示报告"""
    
    report = {
        'svd_configuration': projector.config.to_dict(),
        'projection_info': projector.get_projection_info(),
        'reconstruction_errors': reconstruction_errors,
        'performance_metrics': {
            'input_compression_ratio': projector.input_dim / projector.latent_dim,
            'output_compression_ratio': projector.output_dim / projector.latent_dim,
            'latent_efficiency': projector.latent_dim / max(projector.input_dim, projector.output_dim),
            'total_parameters_saved': (projector.input_dim + projector.output_dim - 2 * projector.latent_dim) / (projector.input_dim + projector.output_dim)
        }
    }
    
    # 保存JSON报告
    report_path = output_dir / 'svd_demo_report.json'
    
    # 兼容numpy类型的JSON编码器，避免 np.int64/np.float32 等无法序列化
    def _np_encoder(o):
        import numpy as _np
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        return str(o)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_np_encoder)
    
    # 生成Markdown摘要
    md_content = f"""# SVD模态分解演示报告

## 配置信息
- **潜在维度**: {projector.latent_dim}
- **输入维度**: {projector.input_dim:,}
- **输出维度**: {projector.output_dim:,}
- **能量阈值**: {projector.config.energy_threshold}
- **自动选择模态**: {projector.config.auto_select_modes}

## 性能指标
- **输入压缩比**: {projector.input_dim / projector.latent_dim:.1f}x
- **输出压缩比**: {projector.output_dim / projector.latent_dim:.1f}x
- **参数节省**: {report['performance_metrics']['total_parameters_saved']:.1%}

## 重建误差
- **输入MSE**: {reconstruction_errors['input_mse']:.6f}
- **输出MSE**: {reconstruction_errors['output_mse']:.6f}
- **输入相对误差**: {reconstruction_errors['input_relative_error']:.4f}
- **输出相对误差**: {reconstruction_errors['output_relative_error']:.4f}

## 结论
SVD模态分解成功将高维数据投影到 {projector.latent_dim} 维潜在空间，实现了有效的维度压缩，
同时保持了较低的重建误差。输入数据压缩比为 {projector.input_dim / projector.latent_dim:.1f}x，
输出数据压缩比为 {projector.output_dim / projector.latent_dim:.1f}x。
"""
    
    md_path = output_dir / 'svd_demo_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"演示报告已保存:")
    logger.info(f"  JSON报告: {report_path}")
    logger.info(f"  Markdown摘要: {md_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SVD模态分解综合演示')
    parser.add_argument('--data-path', type=str, 
                       default=r'x:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5',
                       help='HDF5数据文件路径')
    parser.add_argument('--n-modes', type=int, default=64, help='SVD模态数量')
    parser.add_argument('--max-samples', type=int, default=200, help='最大样本数量')
    parser.add_argument('--output-dir', type=str, default='svd_demo_output', help='输出目录')
    parser.add_argument('--use-synthetic', action='store_true', help='使用合成数据而非HDF5文件')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== SVD模态分解综合演示开始 ===")
    logger.info(f"输出目录: {output_dir}")
    
    try:
        # 加载或创建数据
        if args.use_synthetic:
            logger.info("使用合成数据")
            input_data, output_data = create_synthetic_data(
                n_samples=args.max_samples,
                input_dim=1024,
                output_dim=4096
            )
        else:
            data_path = Path(args.data_path)
            if data_path.exists():
                input_data, output_data = load_hdf5_data(data_path, args.max_samples)
            else:
                logger.warning(f"HDF5文件不存在: {data_path}，使用合成数据")
                input_data, output_data = create_synthetic_data(
                    n_samples=args.max_samples,
                    input_dim=1024,
                    output_dim=4096
                )
        
        # 执行SVD工作流程演示
        projector, reconstruction_errors = demonstrate_svd_workflow(
            input_data, output_data, args.n_modes, output_dir
        )
        
        logger.info("=== 演示完成 ===")
        logger.info(f"结果保存在: {output_dir}")
        logger.info(f"关键发现:")
        logger.info(f"  - 有效模态数: {projector.latent_dim}")
        logger.info(f"  - 输入压缩比: {projector.input_dim / projector.latent_dim:.1f}x")
        logger.info(f"  - 输出压缩比: {projector.output_dim / projector.latent_dim:.1f}x")
        logger.info(f"  - 输入重建相对误差: {reconstruction_errors['input_relative_error']:.4f}")
        logger.info(f"  - 输出重建相对误差: {reconstruction_errors['output_relative_error']:.4f}")
        
    except Exception as e:
        logger.error(f"演示过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()