#!/usr/bin/env python3
"""
输出模态分析工具 - 专门分析神经网络输出的模态结构
支持生成主模态空间基的可视化热图（单张 + 网格汇总）

功能:
1. 从HDF5文件加载输出数据
2. 进行SVD分解分析模态结构
3. 生成可视化图表和分析报告
4. 支持样本相似度分析
5. 生成主模态空间基热图

使用方法:
python analyze_output_modality_only.py --data_path output.h5 --output_dir results/
"""

import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import h5py
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 抑制字体警告
import warnings
warnings.filterwarnings("ignore", message=r".*missing from font.*")
warnings.filterwarnings("ignore", message=r".*does not have a glyph.*")
warnings.filterwarnings("ignore", message=r".*substituting with a dummy symbol.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 中文字体设置
CHINESE_FONT_NAME = None


def configure_chinese_font():
    """配置中文字体支持"""
    global CHINESE_FONT_NAME
    
    try:
        import matplotlib.font_manager as fm
        
        # 候选中文字体
        candidate_fonts = [
            "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
            "Source Han Sans CN", "Noto Sans CJK SC", "PingFang SC",
            "Hiragino Sans GB", "STHeiti", "FangSong", "SimSun"
        ]
        
        # 查找第一个可用的中文字体
        selected = "Arial"
        for font_name in candidate_fonts:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if font_path and os.path.exists(font_path):
                    selected = font_name
                    CHINESE_FONT_NAME = selected
                    break
            except Exception:
                continue
        
        # 配置matplotlib字体
        import matplotlib as mpl
        if selected != "Arial":
            mpl.rcParams["font.sans-serif"] = [selected] + [f for f in candidate_fonts if f != selected]
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["svg.fonttype"] = "none"  # 临时改为 none，让 SVG 直接引用字体
        mpl.rcParams["text.usetex"] = False

        # 触发查找，确保缓存可用
        try:
            fm.findfont(fm.FontProperties(family=selected), rebuild_if_missing=True)
        except Exception:
            pass

        logger.info(
            f"已配置中文字体用于 PNG/SVG: {selected}; 最终使用: {CHINESE_FONT_NAME}; 备选: {mpl.rcParams['font.sans-serif'][:3]}"
        )
    except Exception as e:
        logger.warning(f"字体配置失败: {e}")

# 配置中文字体
configure_chinese_font()
sns.set_style("whitegrid")


class OutputModalityAnalyzer:
    """输出模态分析器 - 仅分析输出数据的模态结构"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_hdf5_dataset(self, data_path: Path) -> Optional[Dict]:
        """加载HDF5数据集"""
        logger.info(f"正在加载HDF5数据集: {data_path}")
        
        if not data_path.exists():
            logger.error(f"数据文件不存在: {data_path}")
            return None
            
        try:
            with h5py.File(data_path, 'r') as f:
                # 检查数据结构
                keys = list(f.keys())
                logger.info(f"HDF5文件包含的数据集: {keys}")
                
                dataset_info = {}
                for key in keys:
                    dataset = f[key]
                    dataset_info[key] = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size_mb': dataset.size * dataset.dtype.itemsize / (1024**2)
                    }
                    logger.info(f"  {key}: 形状={dataset.shape}, 类型={dataset.dtype}, 大小={dataset_info[key]['size_mb']:.2f}MB")
                
                # 主要使用tensor数据集
                if 'tensor' in keys:
                    data = np.array(f['tensor'], dtype=np.float32)
                    logger.info(f"成功加载tensor数据，形状: {data.shape}")
                    
                    return {
                        'data': data,
                        'dataset_info': dataset_info,
                        'file_path': str(data_path)
                    }
                else:
                    logger.error("未找到tensor数据集")
                    return None
                    
        except Exception as e:
            logger.error(f"加载HDF5文件时出错: {str(e)}")
            return None

    def preprocess_output_data(self, data: np.ndarray, max_samples: int = 1000) -> Tuple[np.ndarray, Tuple[int, int]]:
        """预处理输出数据 - 将多维数据展平为2D矩阵，同时返回空间形状(H, W)"""
        logger.info(f"开始预处理数据，原始形状: {data.shape}")
        
        if len(data.shape) == 4:  # [samples, time, height, width]
            n_samples, n_time, H, W = data.shape
            logger.info(f"4D数据: {n_samples}样本, {n_time}时间步, {H}x{W}空间")
            
            # 如果有时间维度，取最后一个时间步作为输出
            if n_time > 1:
                data = data[:, -1, :, :]  # 取最后时间步
                logger.info(f"选择最后时间步，新形状: {data.shape}")
            else:
                data = data[:, 0, :, :]  # 去掉时间维度
                
        elif len(data.shape) == 3:  # [samples, height, width]
            n_samples, H, W = data.shape
            logger.info(f"3D数据: {n_samples}样本, {H}x{W}空间")
            
        else:
            logger.error(f"不支持的数据形状: {data.shape}")
            return None, None
        
        # 随机选择样本
        if n_samples > max_samples:
            indices = np.random.choice(n_samples, max_samples, replace=False)
            data = data[indices]
            n_samples = max_samples
            logger.info(f"随机选择 {max_samples} 个样本")
        
        # 展平空间维度: [samples, H*W]
        output_data = data.reshape(n_samples, -1)
        
        logger.info(f"预处理完成:")
        logger.info(f"  输出数据形状: {output_data.shape}")
        logger.info(f"  空间维度: {H}x{W} = {H*W}")
        
        return output_data, (H, W)

    def perform_svd_analysis(self, output_data: np.ndarray, 
                           n_modes: int = 64,
                           energy_threshold: float = 0.99) -> Dict:
        """进行SVD模态分析"""
        logger.info("开始SVD模态分析...")
        
        n_samples, output_dim = output_data.shape
        logger.info(f"数据维度: 样本数 {n_samples}, 输出维度 {output_dim}")
        
        # 直接使用原始数据进行SVD分解，不进行标准化
        logger.info("计算输出数据SVD...")
        U, S, Vt = np.linalg.svd(output_data, full_matrices=False)
        
        # 自动选择模态数
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        auto_modes = np.argmax(cumulative_energy >= energy_threshold) + 1
        
        # 使用较小的模态数
        effective_modes = min(auto_modes, n_modes, len(S))
        logger.info(f"自动选择模态数: {auto_modes}, 用户指定: {n_modes}, 实际使用: {effective_modes}")
        
        # 计算重建误差
        U_reduced = U[:, :effective_modes]
        S_reduced = S[:effective_modes]
        Vt_reduced = Vt[:effective_modes, :]
        
        # 重建数据
        reconstructed = U_reduced @ np.diag(S_reduced) @ Vt_reduced
        mse = np.mean((output_data - reconstructed)**2)
        relative_error = mse / (np.mean(output_data**2) + 1e-12)
        
        logger.info(f"重建质量: MSE={mse:.6f}, 相对误差={relative_error:.6f}")
        
        # 计算样本的嵌入向量（降维表示）
        sample_embeddings = U_reduced @ np.diag(S_reduced)
        
        analysis_results = {
            'singular_values': S,
            'energy_ratios': (S**2) / np.sum(S**2),
            'cumulative_energy': cumulative_energy,
            'effective_modes': effective_modes,
            'output_dim': output_dim,
            'n_samples': n_samples,
            'sample_embeddings': sample_embeddings,  # 新增：样本嵌入向量
            'Vt_reduced': Vt_reduced,               # 新增：用于可视化主模态空间基
            'reconstruction_error': {
                'mse': float(mse),
                'relative_error': float(relative_error)
            },
            'config': {
                'n_modes': n_modes,
                'energy_threshold': energy_threshold,
                'auto_selected_modes': int(auto_modes)
            }
        }
        
        return analysis_results

    def perform_per_sample_svd_analysis(self, output_data: np.ndarray, spatial_shape: Tuple[int, int],
                                      n_modes: int = 64,
                                      energy_threshold: float = 0.99) -> Dict:
        """对每个样本的H×W矩阵单独进行SVD，然后对所有样本的右奇异向量进行平均"""
        logger.info("开始逐样本SVD模态分析...")
        
        n_samples, output_dim = output_data.shape
        H, W = spatial_shape
        
        if output_dim != H * W:
            logger.error(f"数据维度不匹配: output_dim={output_dim}, H*W={H*W}")
            return {}
        
        logger.info(f"数据维度: 样本数 {n_samples}, 空间维度 {H}×{W}")
        
        # 将展平的数据重新塑形为[N, H, W]
        reshaped_data = output_data.reshape(n_samples, H, W)
        
        # 对每个样本进行SVD
        all_Vt = []
        all_singular_values = []
        sample_embeddings_list = []
        # 新增：保存每个样本的U前k列，用于构建H×W模态基的平均图像
        all_U = []
        
        effective_modes = min(n_modes, H, W)  # 确保模态数不超过矩阵维度
        
        for i in range(n_samples):
            sample_matrix = reshaped_data[i]  # [H, W]
            
            # 对单个样本的H×W矩阵进行SVD
            U_sample, S_sample, Vt_sample = np.linalg.svd(sample_matrix, full_matrices=False)
            
            # 根据能量阈值自动选择模态数
            cumulative_energy = np.cumsum(S_sample**2) / np.sum(S_sample**2) if len(S_sample) > 0 else np.array([1.0])
            auto_modes = np.argmax(cumulative_energy >= energy_threshold) + 1 if len(cumulative_energy) > 0 else 1
            sample_effective_modes = min(auto_modes, effective_modes, len(S_sample))
            
            # 截取有效模态
            Vt_sample_reduced = Vt_sample[:sample_effective_modes, :]
            S_sample_reduced = S_sample[:sample_effective_modes]
            U_sample_reduced = U_sample[:, :sample_effective_modes]
            
            # 符号对齐：确保第一个模态的第一个元素为正
            for mode_idx in range(sample_effective_modes):
                if Vt_sample_reduced[mode_idx, 0] < 0:
                    Vt_sample_reduced[mode_idx, :] *= -1
                    U_sample_reduced[:, mode_idx] *= -1
            
            # 补齐到统一维度
            if sample_effective_modes < effective_modes:
                padding_Vt = np.zeros((effective_modes - sample_effective_modes, W))
                Vt_sample_reduced = np.vstack([Vt_sample_reduced, padding_Vt])
                
                padding_S = np.zeros(effective_modes - sample_effective_modes)
                S_sample_reduced = np.concatenate([S_sample_reduced, padding_S])
                
                padding_U = np.zeros((H, effective_modes - sample_effective_modes))
                U_sample_reduced = np.hstack([U_sample_reduced, padding_U])
            
            all_Vt.append(Vt_sample_reduced)
            all_singular_values.append(S_sample_reduced)
            # 保存 U 的前k列
            all_U.append(U_sample_reduced)
            
            # 计算样本嵌入（U @ S的前k个模态）
            sample_embedding = (U_sample_reduced @ np.diag(S_sample_reduced))[:, :effective_modes]
            sample_embeddings_list.append(sample_embedding.flatten())  # 展平为1D
        
        # 将所有样本的Vt矩阵进行平均
        all_Vt = np.array(all_Vt)  # [N, effective_modes, W]
        all_singular_values = np.array(all_singular_values)  # [N, effective_modes]
        all_U = np.array(all_U)  # [N, H, effective_modes]
        
        # 对右奇异向量进行平均（宽度方向）
        averaged_Vt_width = np.mean(all_Vt, axis=0)  # [effective_modes, W]
        averaged_singular_values = np.mean(all_singular_values, axis=0)  # [effective_modes]
        
        # 计算平均后的能量比和累积能量
        energy_ratios = (averaged_singular_values**2) / np.sum(averaged_singular_values**2) if np.sum(averaged_singular_values**2) > 0 else np.zeros_like(averaged_singular_values)
        cumulative_energy = np.cumsum(energy_ratios)
        
        # 构建样本嵌入矩阵
        sample_embeddings_array = np.array(sample_embeddings_list)  # [N, H*effective_modes]
        
        # 使用平均后的Vt（宽度方向）重建数据以计算误差
        reconstructed_samples = []
        for i in range(n_samples):
            # 使用原始样本的U和S，但是平均后的Vt进行重建
            sample_matrix = reshaped_data[i]
            U_sample, S_sample, _ = np.linalg.svd(sample_matrix, full_matrices=False)
            
            # 只使用前effective_modes个模态
            U_reduced = U_sample[:, :effective_modes]
            S_reduced = S_sample[:effective_modes]
            
            # 重建使用平均后的Vt（宽度方向平均）
            reconstructed = U_reduced @ np.diag(S_reduced) @ averaged_Vt_width
            reconstructed_samples.append(reconstructed.flatten())
        
        # 计算重建误差
        original_flat = output_data  # [N, H*W]
        reconstructed_flat = np.array(reconstructed_samples)  # [N, H*W]
        
        mse = np.mean((original_flat - reconstructed_flat)**2)
        relative_error = mse / (np.mean(original_flat**2) + 1e-12)
        
        # 计算 H×W 的平均模态基（按奇异值加权的外积平均）
        averaged_mode_images = np.zeros((effective_modes, H, W), dtype=np.float64)
        for m in range(effective_modes):
            accum = np.zeros((H, W), dtype=np.float64)
            weight_sum = 0.0
            for i in range(n_samples):
                ui = all_U[i, :, m]  # [H]
                vi = all_Vt[i, m, :]  # [W]
                si = all_singular_values[i, m]
                if si > 0:
                    accum += si * np.outer(ui, vi)
                    weight_sum += si
            if weight_sum > 0:
                averaged_mode_images[m] = accum / weight_sum
            else:
                averaged_mode_images[m] = np.mean([np.outer(all_U[i, :, m], all_Vt[i, m, :]) for i in range(n_samples)], axis=0)
        
        # 展平成与标准SVD一致的形状 [k, H*W]，便于可视化和投影
        Vt_reduced_hw = averaged_mode_images.reshape(effective_modes, H * W)
        
        logger.info(f"逐样本SVD完成:")
        logger.info(f"  有效模态数: {effective_modes}")
        logger.info(f"  平均重建误差: MSE={mse:.6f}, 相对误差={relative_error:.6f}")
        
        analysis_results = {
            'singular_values': averaged_singular_values,
            'energy_ratios': energy_ratios,
            'cumulative_energy': cumulative_energy,
            'effective_modes': effective_modes,
            'output_dim': output_dim,
            'n_samples': n_samples,
            'sample_embeddings': sample_embeddings_array,
            'Vt_reduced': Vt_reduced_hw,  # [k, H*W]，用于可视化
            'Vt_width_averaged': averaged_Vt_width,  # [k, W]，保留参考
            'reconstruction_error': {
                'mse': float(mse),
                'relative_error': float(relative_error)
            },
            'config': {
                'n_modes': n_modes,
                'energy_threshold': energy_threshold,
                'method': 'per_sample_averaged'
            },
            'per_sample_stats': {
                'individual_singular_values': all_singular_values.tolist(),
                'individual_Vt_shapes': [vt.shape for vt in all_Vt],
                'avg_energy_per_mode': energy_ratios.tolist() if isinstance(energy_ratios, np.ndarray) else energy_ratios
            },
            # 新增：保存逐样本SVD的详细数据用于专门的系数分析
            'per_sample_svd_data': {
                'all_singular_values': all_singular_values,  # [N, effective_modes]
                'all_U': all_U,  # [N, H, effective_modes] 
                'all_Vt': all_Vt,  # [N, effective_modes, W]
                'spatial_shape': spatial_shape,
                'reshaped_data': reshaped_data  # [N, H, W]
            }
        }
        
        return analysis_results

    def visualize_modal_bases(self, Vt_reduced: np.ndarray, spatial_shape: Tuple[int, int], max_modes_to_save: int = 9) -> Dict[str, object]:
        """可视化主模态的空间基（单张 + 网格汇总）并返回保存路径"""
        try:
            H, W = int(spatial_shape[0]), int(spatial_shape[1])
        except Exception:
            logger.error(f"非法的空间形状: {spatial_shape}")
            return {}
            
        m = Vt_reduced.shape[0]
        k = min(m, max_modes_to_save)
        single_png_paths: List[str] = []
        single_svg_paths: List[str] = []
        
        # 单模态热图
        for i in range(k):
            mode_vec = Vt_reduced[i]
            mode_img = mode_vec.reshape(H, W)
            vmax = np.max(np.abs(mode_img)) + 1e-12
            vmin = -vmax
            
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            im = ax.imshow(mode_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.set_title(f'Mode {i+1}')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Amplitude', rotation=90)
            
            fname_base = f'modal_heatmap_mode_{i+1:02d}'
            png_path = self.output_dir / f'{fname_base}.png'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            
            # 保存SVG格式
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                svg_path = self.output_dir / f'{fname_base}.svg'
                plt.savefig(svg_path, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                pass
            
            plt.close(fig)
            single_png_paths.append(str(png_path))
            single_svg_paths.append(str(svg_path))
        
        logger.info(f"已导出 {len(single_png_paths)} 张单模态热图")
        
        # 网格汇总
        cols = 3
        rows = int(np.ceil(k / cols)) if k > 0 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4))
        
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        axes = axes.reshape(rows, cols)
        
        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            ax.axis('off')
            
            if idx < k:
                mode_img = Vt_reduced[idx].reshape(H, W)
                vmax = np.max(np.abs(mode_img)) + 1e-12
                vmin = -vmax
                im = ax.imshow(mode_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.set_title(f'Mode {idx+1}')
        
        plt.tight_layout()
        
        # 强制中文字体
        try:
            import matplotlib
            fam = CHINESE_FONT_NAME
            if fam:
                for ax in axes.ravel():
                    for txt in ax.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
        except Exception:
            pass
        
        grid_png = self.output_dir / 'modal_heatmaps_grid_summary.png'
        plt.savefig(grid_png, dpi=300, bbox_inches='tight')
        
        try:
            import matplotlib as mpl
            old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
            mpl.rcParams['svg.fonttype'] = 'none'
            grid_svg = self.output_dir / 'modal_heatmaps_grid_summary.svg'
            plt.savefig(grid_svg, format='svg', bbox_inches='tight')
            if old_svg_fonttype is not None:
                mpl.rcParams['svg.fonttype'] = old_svg_fonttype
        except Exception:
            pass
        
        plt.close(fig)
        logger.info(f"网格汇总图已保存到: {grid_png}")
        
        return {
            'single_png': single_png_paths,
            'single_svg': single_svg_paths,
            'grid_png': str(grid_png),
            'grid_svg': str(grid_svg),
        }

    def visualize_analysis(self, analysis_results: Dict):
        """生成分析可视化图表"""
        logger.info("开始生成可视化图表...")
        
        # 提取数据
        S = analysis_results['singular_values']
        energy_ratios = analysis_results['energy_ratios']
        cumulative_energy = analysis_results['cumulative_energy']
        effective_modes = analysis_results['effective_modes']
        config = analysis_results['config']
        
        # 创建综合图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 奇异值分布
        ax1.semilogy(range(1, min(len(S)+1, 101)), S[:100], 'b-o', markersize=3)
        ax1.axvline(effective_modes, color='red', linestyle='--', label=f'有效模态数: {effective_modes}')
        ax1.set_xlabel('模态索引')
        ax1.set_ylabel('奇异值 (对数尺度)')
        ax1.set_title('奇异值分布')
        ax1.grid(True)
        ax1.legend()
        
        # 2. 能量比例
        ax2.bar(range(1, min(len(energy_ratios)+1, 21)), energy_ratios[:20], color='skyblue')
        ax2.set_xlabel('模态索引')
        ax2.set_ylabel('能量比例')
        ax2.set_title('前20个模态的能量比例')
        ax2.grid(True)
        
        # 3. 累积能量
        ax3.plot(range(1, len(cumulative_energy)+1), cumulative_energy, 'g-', linewidth=2)
        ax3.axhline(config['energy_threshold'], color='red', linestyle='--', 
                   label=f'阈值: {config["energy_threshold"]:.2f}')
        ax3.axvline(effective_modes, color='red', linestyle='--', 
                   label=f'有效模态: {effective_modes}')
        ax3.set_xlabel('模态数量')
        ax3.set_ylabel('累积能量比例')
        ax3.set_title('累积能量比例')
        ax3.grid(True)
        ax3.legend()
        
        # 4. 统计信息
        ax4.axis('off')
        stats_text = f"""
模态分析统计信息

数据维度: {analysis_results['n_samples']} × {analysis_results['output_dim']}
有效模态数: {effective_modes}
压缩比: {analysis_results['output_dim'] / effective_modes:.1f}x

重建误差:
- MSE: {analysis_results['reconstruction_error']['mse']:.6f}
- 相对误差: {analysis_results['reconstruction_error']['relative_error']:.6f}

配置参数:
- 指定模态数: {config['n_modes']}
- 能量阈值: {config['energy_threshold']:.2f}
- 自动选择模态数: {config['auto_selected_modes']}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        # 设置中文字体
        try:
            fam = CHINESE_FONT_NAME
            if fam:
                for ax in [ax1, ax2, ax3]:
                    for txt in ax.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
        except Exception:
            pass
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.output_dir / "modal_analysis_output.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # 保存SVG格式
        try:
            import matplotlib as mpl
            old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
            mpl.rcParams['svg.fonttype'] = 'none'
            svg_output_path = self.output_dir / "modal_analysis_output.svg"
            plt.savefig(svg_output_path, format='svg', bbox_inches='tight')
            if old_svg_fonttype is not None:
                mpl.rcParams['svg.fonttype'] = old_svg_fonttype
        except Exception:
            pass
        
        logger.info(f"可视化图表已保存到: {output_path}")
        plt.close()
        return output_path

    def save_similarity_matrices(self, similarities: Dict) -> Dict[str, str]:
        """保存相似度矩阵到文件"""
        paths = {}
        for key, matrix in similarities.items():
            path = self.output_dir / f"similarity_{key}.npy"
            np.save(path, matrix)
            paths[key] = str(path)
        return paths

    def summarize_similarities(self, cosine_sim: np.ndarray, top_k: int = 5) -> Dict:
        """生成相似度摘要统计"""
        # 去除对角线元素
        mask = ~np.eye(cosine_sim.shape[0], dtype=bool)
        valid_similarities = cosine_sim[mask]
        
        summary = {
            'avg': float(np.mean(valid_similarities)),
            'std': float(np.std(valid_similarities)),
            'min': float(np.min(valid_similarities)),
            'max': float(np.max(valid_similarities)),
            'median': float(np.median(valid_similarities))
        }
        
        return summary

    def compute_sample_similarities(self, embeddings: np.ndarray) -> Dict:
        """计算样本间的相似度矩阵"""
        logger.info("计算样本间相似度...")
        
        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        cosine_sim = cosine_similarity(embeddings)
        euclidean_dist = euclidean_distances(embeddings)
        
        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist
        }

    def visualize_sample_similarities(self, similarities: Dict, sample_count: int, max_show: int = 200) -> Dict[str, Path]:
        """可视化样本相似度/距离热力图（同时展示余弦相似度与欧氏距离）"""
        logger.info("生成相似度热力图(余弦 + 欧氏距离)...")
        
        cosine_sim = similarities['cosine_similarity']
        euclidean_dist = similarities.get('euclidean_distance', None)
        
        # 如果样本太多，随机选择一部分显示（对两种度量使用同一索引）
        if sample_count > max_show:
            indices = np.random.choice(sample_count, max_show, replace=False)
            indices = np.sort(indices)
            cosine_sim_display = cosine_sim[np.ix_(indices, indices)]
            euclidean_display = euclidean_dist[np.ix_(indices, indices)] if euclidean_dist is not None else None
            title_suffix = f" (随机显示 {max_show}/{sample_count} 样本)"
        else:
            cosine_sim_display = cosine_sim
            euclidean_display = euclidean_dist
            title_suffix = f" ({sample_count} 样本)"
        
        output_paths: Dict[str, Path] = {}
        
        # 1) 单独保存：余弦相似度热力图（保持兼容）
        plt.figure(figsize=(12, 10))
        font_size = max(6, min(12, 200 // cosine_sim_display.shape[0]))
        sns.heatmap(
            cosine_sim_display,
            cmap='RdYlBu_r',
            center=0,
            square=True,
            vmin=-1, vmax=1,
            cbar_kws={'label': '余弦相似度'},
            annot=False,
            fmt='.2f'
        )
        plt.title(f'样本间余弦相似度热力图{title_suffix}')
        plt.xlabel('样本索引')
        plt.ylabel('样本索引')
        try:
            fam = CHINESE_FONT_NAME
            if fam:
                plt.gca().set_title(f'样本间余弦相似度热力图{title_suffix}', fontfamily=fam)
                plt.gca().set_xlabel('样本索引', fontfamily=fam)
                plt.gca().set_ylabel('样本索引', fontfamily=fam)
        except Exception:
            pass
        plt.tight_layout()
        png_path = self.output_dir / "sample_similarity_heatmap.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        try:
            import matplotlib as mpl
            old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
            mpl.rcParams['svg.fonttype'] = 'none'
            svg_path = self.output_dir / "sample_similarity_heatmap.svg"
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            if old_svg_fonttype is not None:
                mpl.rcParams['svg.fonttype'] = old_svg_fonttype
        except Exception:
            svg_path = png_path
        plt.close()
        output_paths['cosine_png'] = png_path
        output_paths['cosine_svg'] = svg_path
        logger.info(f"余弦相似度热力图已保存到: {png_path}")
        
        # 2) 单独保存：欧氏距离热力图
        if euclidean_display is not None:
            plt.figure(figsize=(12, 10))
            font_size = max(6, min(12, 200 // euclidean_display.shape[0]))
            sns.heatmap(
                euclidean_display,
                cmap='viridis',
                square=True,
                vmin=0, vmax=float(np.nanmax(euclidean_display)),
                cbar_kws={'label': '欧氏距离'},
                annot=False,
                fmt='.2f'
            )
            plt.title(f'样本间欧氏距离热力图{title_suffix}')
            plt.xlabel('样本索引')
            plt.ylabel('样本索引')
            try:
                fam = CHINESE_FONT_NAME
                if fam:
                    plt.gca().set_title(f'样本间欧氏距离热力图{title_suffix}', fontfamily=fam)
                    plt.gca().set_xlabel('样本索引', fontfamily=fam)
                    plt.gca().set_ylabel('样本索引', fontfamily=fam)
            except Exception:
                pass
            plt.tight_layout()
            e_png_path = self.output_dir / "sample_euclidean_distance_heatmap.png"
            plt.savefig(e_png_path, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                e_svg_path = self.output_dir / "sample_euclidean_distance_heatmap.svg"
                plt.savefig(e_svg_path, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                e_svg_path = e_png_path
            plt.close()
            output_paths['euclidean_png'] = e_png_path
            output_paths['euclidean_svg'] = e_svg_path
            logger.info(f"欧氏距离热力图已保存到: {e_png_path}")
        
        # 3) 组合图（左：余弦相似度；右：欧氏距离）
        try:
            if euclidean_display is not None:
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                ax1, ax2 = axes
                sns.heatmap(
                    cosine_sim_display,
                    ax=ax1,
                    cmap='RdYlBu_r',
                    center=0,
                    square=True,
                    vmin=-1, vmax=1,
                    cbar_kws={'label': '余弦相似度'},
                    annot=False
                )
                ax1.set_title(f'余弦相似度{title_suffix}')
                ax1.set_xlabel('样本索引')
                ax1.set_ylabel('样本索引')
                sns.heatmap(
                    euclidean_display,
                    ax=ax2,
                    cmap='viridis',
                    square=True,
                    vmin=0, vmax=float(np.nanmax(euclidean_display)),
                    cbar_kws={'label': '欧氏距离'},
                    annot=False
                )
                ax2.set_title(f'欧氏距离{title_suffix}')
                ax2.set_xlabel('样本索引')
                ax2.set_ylabel('样本索引')
                try:
                    fam = CHINESE_FONT_NAME
                    if fam:
                        ax1.set_title(f'余弦相似度{title_suffix}', fontfamily=fam)
                        ax1.set_xlabel('样本索引', fontfamily=fam)
                        ax1.set_ylabel('样本索引', fontfamily=fam)
                        ax2.set_title(f'欧氏距离{title_suffix}', fontfamily=fam)
                        ax2.set_xlabel('样本索引', fontfamily=fam)
                        ax2.set_ylabel('样本索引', fontfamily=fam)
                except Exception:
                    pass
                plt.suptitle('样本间模态相似度/距离对比', fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                combo_png = self.output_dir / "output_sample_similarity.png"
                plt.savefig(combo_png, dpi=300, bbox_inches='tight')
                try:
                    import matplotlib as mpl
                    old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                    mpl.rcParams['svg.fonttype'] = 'none'
                    combo_svg = self.output_dir / "output_sample_similarity.svg"
                    plt.savefig(combo_svg, format='svg', bbox_inches='tight')
                    if old_svg_fonttype is not None:
                        mpl.rcParams['svg.fonttype'] = old_svg_fonttype
                except Exception:
                    combo_svg = combo_png
                plt.close()
                output_paths['combined_png'] = combo_png
                output_paths['combined_svg'] = combo_svg
                logger.info(f"相似度/距离组合图已保存到: {combo_png}")
        except Exception as e:
            logger.warning(f"生成组合相似度图失败: {e}")
        
        return output_paths

    def analyze_per_mode_sample_similarity(self, sample_embeddings: np.ndarray, 
                                         n_modes: int = None, 
                                         max_show: int = 200) -> Dict[str, object]:
        """分析每个模态下样本间的相似度并生成热力图
        
        Args:
            sample_embeddings: 样本嵌入矩阵 [n_samples, k], 即 U_k @ diag(S_k)
            n_modes: 要分析的模态数，默认为 sample_embeddings 的列数
            max_show: 每张热力图最多显示的样本数
            
        Returns:
            包含每个模态热力图路径的字典
        """
        logger.info("开始每个模态的样本相似度分析...")
        
        n_samples, k_total = sample_embeddings.shape
        if n_modes is None:
            n_modes = k_total
        else:
            n_modes = min(n_modes, k_total)
        
        # 如果样本太多，随机选择一部分显示
        if n_samples > max_show:
            indices = np.random.choice(n_samples, max_show, replace=False)
            indices = np.sort(indices)
            embeddings_display = sample_embeddings[indices]
            title_suffix = f" (随机显示 {max_show}/{n_samples} 样本)"
        else:
            embeddings_display = sample_embeddings
            title_suffix = f" ({n_samples} 样本)"
        
        output_paths = {}
        
        # 为每个模态生成样本相似度热力图
        for mode_idx in range(n_modes):
            # 提取第 mode_idx 个模态的系数向量：每个样本在该模态上的投影
            mode_coeffs = embeddings_display[:, mode_idx].reshape(-1, 1)  # [n_samples_display, 1]
            
            # 计算该模态下样本间的相似度
            from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
            cosine_sim = cosine_similarity(mode_coeffs)
            euclidean_dist = euclidean_distances(mode_coeffs)
            
            # 生成该模态的相似度热力图
            plt.figure(figsize=(10, 8))
            
            # 创建热力图前确保中文字体设置已生效
            try:
                import matplotlib as mpl
                # 确保中文字体设置
                current_font_family = mpl.rcParams.get('font.family', ['sans-serif'])
                current_sans_serif = mpl.rcParams.get('font.sans-serif', ['DejaVu Sans'])
                
                # 临时设置中文字体（如果CHINESE_FONT_NAME已定义）
                if CHINESE_FONT_NAME:
                    mpl.rcParams['font.family'] = 'sans-serif'
                    mpl.rcParams['font.sans-serif'] = [CHINESE_FONT_NAME] + [f for f in current_sans_serif if f != CHINESE_FONT_NAME]
                    
            except Exception:
                pass
            
            sns.heatmap(
                cosine_sim,
                cmap='RdYlBu_r',
                center=0,
                square=True,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'cosine similarity'},  # 使用英文避免字体问题
                annot=False,
                fmt='.3f'
            )
            
            # 使用英文标题避免字体问题
            plt.title(f'Mode {mode_idx+1} Sample Cosine Similarity{title_suffix}')
            plt.xlabel('Sample Index')
            plt.ylabel('Sample Index')
            
            plt.tight_layout()
            
            # 保存PNG格式
            png_path = self.output_dir / f"per_mode_similarity_mode_{mode_idx+1:02d}.png"
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            
            # 保存SVG格式  
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'path'  # 使用路径化避免字体问题
                svg_path = self.output_dir / f"per_mode_similarity_mode_{mode_idx+1:02d}.svg"
                plt.savefig(svg_path, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                svg_path = png_path
            
            plt.close()
            
            # 记录路径和统计信息
            output_paths[f'mode_{mode_idx+1}_png'] = png_path
            output_paths[f'mode_{mode_idx+1}_svg'] = svg_path
            output_paths[f'mode_{mode_idx+1}_cosine_stats'] = {
                'mean': float(np.mean(cosine_sim)),
                'std': float(np.std(cosine_sim)),
                'min': float(np.min(cosine_sim)),
                'max': float(np.max(cosine_sim))
            }
            output_paths[f'mode_{mode_idx+1}_euclidean_stats'] = {
                'mean': float(np.mean(euclidean_dist)),
                'std': float(np.std(euclidean_dist)),
                'min': float(np.min(euclidean_dist)),
                'max': float(np.max(euclidean_dist))
            }
            
            logger.info(f"模态 {mode_idx+1} 相似度热力图已保存到: {png_path}")
        
        # 生成汇总报告
        summary_info = {
            'analyzed_modes': n_modes,
            'total_samples': n_samples,
            'displayed_samples': embeddings_display.shape[0],
            'heatmaps_generated': n_modes
        }
        output_paths['summary'] = summary_info
        
        logger.info(f"共生成 {n_modes} 张每模态样本相似度热力图")
        return output_paths
    
    def visualize_coefficients(self, coefficients: np.ndarray, n_modes: int = None) -> Dict[str, str]:
        """生成系数的可视化图表：均值柱状图、分布箱线图和热力图
        
        Args:
            coefficients: 系数矩阵 [n_samples, k_modes]
            n_modes: 要可视化的模态数，默认为所有模态
            
        Returns:
            包含可视化文件路径的字典
        """
        logger.info("开始生成系数可视化图表...")
        
        n_samples, k_total = coefficients.shape
        if n_modes is None:
            n_modes = k_total
        else:
            n_modes = min(n_modes, k_total)
        
        # 只使用前 n_modes 个模态
        coeffs_display = coefficients[:, :n_modes]
        
        output_paths = {}
        
        # 1) 系数均值柱状图
        try:
            coeff_means = np.mean(coeffs_display, axis=0)
            coeff_stds = np.std(coeffs_display, axis=0)
            
            plt.figure(figsize=(12, 6))
            mode_indices = np.arange(n_modes)
            bars = plt.bar(mode_indices, coeff_means, yerr=coeff_stds, 
                          alpha=0.7, color='steelblue', edgecolor='black', capsize=5)
            
            plt.xlabel('模态索引')
            plt.ylabel('系数均值')
            plt.title(f'各模态系数均值 (±标准差, {n_samples}个样本)')
            plt.xticks(mode_indices, [f'模态{i+1}' for i in range(n_modes)], rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (mean, std) in enumerate(zip(coeff_means, coeff_stds)):
                plt.text(i, mean + std + 0.01 * (coeff_means.max() - coeff_means.min()), 
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 字体设置
            try:
                fam = CHINESE_FONT_NAME
                if fam:
                    plt.gca().set_xlabel('模态索引', fontfamily=fam)
                    plt.gca().set_ylabel('系数均值', fontfamily=fam)
                    plt.gca().set_title(f'各模态系数均值 (±标准差, {n_samples}个样本)', fontfamily=fam)
            except Exception:
                pass
            
            plt.tight_layout()
            mean_png = self.output_dir / 'coefficients_mean_bar.png'
            plt.savefig(mean_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                mean_svg = self.output_dir / 'coefficients_mean_bar.svg'
                plt.savefig(mean_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                mean_svg = mean_png
            plt.close()
            
            output_paths['mean_bar_png'] = str(mean_png)
            output_paths['mean_bar_svg'] = str(mean_svg)
            logger.info(f"系数均值柱状图已保存到: {mean_png}")
            
        except Exception as e:
            logger.warning(f"生成系数均值柱状图失败: {e}")
        
        # 2) 系数分布箱线图
        try:
            plt.figure(figsize=(12, 8))
            box_data = [coeffs_display[:, i] for i in range(n_modes)]
            box_labels = [f'模态{i+1}' for i in range(n_modes)]
            
            bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True, 
                           notch=True, showmeans=True, meanline=True)
            
            # 设置箱线图颜色
            colors = plt.cm.Set3(np.linspace(0, 1, n_modes))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.xlabel('模态')
            plt.ylabel('系数值')
            plt.title(f'各模态系数分布箱线图 ({n_samples}个样本)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 字体设置
            try:
                fam = CHINESE_FONT_NAME
                if fam:
                    plt.gca().set_xlabel('模态', fontfamily=fam)
                    plt.gca().set_ylabel('系数值', fontfamily=fam)
                    plt.gca().set_title(f'各模态系数分布箱线图 ({n_samples}个样本)', fontfamily=fam)
            except Exception:
                pass
            
            plt.tight_layout()
            box_png = self.output_dir / 'coefficients_boxplot.png'
            plt.savefig(box_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                box_svg = self.output_dir / 'coefficients_boxplot.svg'
                plt.savefig(box_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                box_svg = box_png
            plt.close()
            
            output_paths['boxplot_png'] = str(box_png)
            output_paths['boxplot_svg'] = str(box_svg)
            logger.info(f"系数分布箱线图已保存到: {box_png}")
            
        except Exception as e:
            logger.warning(f"生成系数分布箱线图失败: {e}")
        
        # 3) 系数相关性热力图
        try:
            # 计算模态间系数的相关性
            corr_matrix = np.corrcoef(coeffs_display.T)  # [n_modes, n_modes]
            
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角
            
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', mask=mask,
                       cmap='RdBu_r', center=0, square=True,
                       cbar_kws={'label': '相关系数'},
                       xticklabels=[f'模态{i+1}' for i in range(n_modes)],
                       yticklabels=[f'模态{i+1}' for i in range(n_modes)])
            
            plt.title(f'模态间系数相关性热力图 ({n_samples}个样本)')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # 字体设置
            try:
                fam = CHINESE_FONT_NAME
                if fam:
                    plt.gca().set_title(f'模态间系数相关性热力图 ({n_samples}个样本)', fontfamily=fam)
            except Exception:
                pass
            
            plt.tight_layout()
            corr_png = self.output_dir / 'coefficients_correlation_heatmap.png'
            plt.savefig(corr_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                corr_svg = self.output_dir / 'coefficients_correlation_heatmap.svg'
                plt.savefig(corr_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                corr_svg = corr_png
            plt.close()
            
            output_paths['correlation_png'] = str(corr_png)
            output_paths['correlation_svg'] = str(corr_svg)
            logger.info(f"系数相关性热力图已保存到: {corr_png}")
            
        except Exception as e:
            logger.warning(f"生成系数相关性热力图失败: {e}")
        
        logger.info(f"系数可视化完成，共生成 {len(output_paths)} 个文件")
        return output_paths

    def analyze_per_sample_svd_coefficients(self, analysis_results: Dict) -> Dict[str, str]:
        """专门分析逐样本SVD的系数分布、模态矩阵均值及加权均态重构"""
        if 'per_sample_svd_data' not in analysis_results:
            logger.warning("未找到逐样本SVD数据，跳过系数分析")
            return {}
            
        per_sample_data = analysis_results['per_sample_svd_data']
        all_singular_values = per_sample_data['all_singular_values']  # [N, effective_modes]
        all_U = per_sample_data['all_U']  # [N, H, effective_modes]
        all_Vt = per_sample_data['all_Vt']  # [N, effective_modes, W]
        spatial_shape = per_sample_data['spatial_shape']
        reshaped_data = per_sample_data['reshaped_data']  # [N, H, W]
        
        H, W = spatial_shape
        N, effective_modes = all_singular_values.shape
        output_paths = {}
        
        logger.info(f"开始逐样本SVD系数与模态矩阵分析 (N={N}, modes={effective_modes}, H×W={H}×{W})")
        
        # 1. 每个样本的模态系数分布（奇异值即系数）
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1.1 所有样本系数均值的模态分布
            coeff_means = np.mean(all_singular_values, axis=0)  # [effective_modes]
            coeff_stds = np.std(all_singular_values, axis=0)
            axes[0,0].bar(range(effective_modes), coeff_means, yerr=coeff_stds, 
                         alpha=0.7, color='steelblue', capsize=3)
            axes[0,0].set_title('各模态系数均值 (±标准差)')
            axes[0,0].set_xlabel('模态索引')
            axes[0,0].set_ylabel('系数均值')
            axes[0,0].grid(True, alpha=0.3)
            
            # 1.2 系数热力图：样本 × 模态
            display_samples = min(N, 100)
            if N > display_samples:
                sample_indices = np.linspace(0, N-1, display_samples, dtype=int)
                display_coeffs = all_singular_values[sample_indices]
            else:
                display_coeffs = all_singular_values
                sample_indices = np.arange(N)
            
            im = axes[0,1].imshow(display_coeffs, cmap='viridis', aspect='auto')
            axes[0,1].set_title(f'样本-模态系数热力图 ({display_samples}×{effective_modes})')
            axes[0,1].set_xlabel('模态索引')
            axes[0,1].set_ylabel('样本索引')
            plt.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)
            
            # 1.3 系数分布的箱线图（前8个模态）
            n_box = min(8, effective_modes)
            box_data = [all_singular_values[:, i] for i in range(n_box)]
            bp = axes[1,0].boxplot(box_data, labels=[f'M{i+1}' for i in range(n_box)], patch_artist=True)
            for patch, color in zip(bp['boxes'], plt.cm.tab10(np.linspace(0, 1, n_box))):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[1,0].set_title(f'前{n_box}个模态系数分布')
            axes[1,0].set_xlabel('模态')
            axes[1,0].set_ylabel('系数值')
            axes[1,0].grid(True, alpha=0.3)
            
            # 1.4 系数相关性分析
            corr_matrix = np.corrcoef(all_singular_values.T)  # [modes, modes]
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdBu_r', center=0, ax=axes[1,1],
                       xticklabels=[f'M{i+1}' for i in range(effective_modes)],
                       yticklabels=[f'M{i+1}' for i in range(effective_modes)])
            axes[1,1].set_title('模态间系数相关性')
            
            # 字体设置
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass
            
            plt.tight_layout()
            coeff_analysis_png = self.output_dir / 'per_sample_svd_coefficients_analysis.png'
            plt.savefig(coeff_analysis_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                coeff_analysis_svg = self.output_dir / 'per_sample_svd_coefficients_analysis.svg'
                plt.savefig(coeff_analysis_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                coeff_analysis_svg = coeff_analysis_png
            plt.close(fig)
            
            output_paths['coefficients_analysis_png'] = str(coeff_analysis_png)
            output_paths['coefficients_analysis_svg'] = str(coeff_analysis_svg)
            logger.info(f"系数分析图已保存到: {coeff_analysis_png}")
            
        except Exception as e:
            logger.warning(f"生成系数分析失败: {e}")
        
        # 2. 每个模态的平均矩阵（所有样本的模态矩阵均值，直接对样本外积取均值；并提供按系数加权均值与标准差）
        try:
            # 对齐符号：以第一个样本的U[:,m]作为参考，保证外积方向一致，避免正负抵消
            sign_aligned_U = np.copy(all_U)
            sign_aligned_Vt = np.copy(all_Vt)
            for m in range(effective_modes):
                ref_u = all_U[0, :, m]
                ref_norm = np.linalg.norm(ref_u) + 1e-12
                for i in range(N):
                    sgn = 1.0
                    dot = float(np.dot(all_U[i, :, m], ref_u) / ref_norm)
                    if dot < 0:
                        sgn = -1.0
                    sign_aligned_U[i, :, m] *= sgn
                    sign_aligned_Vt[i, m, :] *= sgn
            
            # 直接构造每个样本的模态矩阵 M_i^m = U_i[:,m] ⊗ V_i[m,:]
            mode_matrices = np.zeros((effective_modes, N, H, W), dtype=np.float64)
            for m in range(effective_modes):
                for i in range(N):
                    ui = sign_aligned_U[i, :, m]
                    vi = sign_aligned_Vt[i, m, :]
                    mode_matrices[m, i] = np.outer(ui, vi)
            
            # 2.1 非加权均值与标准差
            mean_unweighted = np.mean(mode_matrices, axis=1)  # [modes, H, W]
            std_unweighted = np.std(mode_matrices, axis=1)    # [modes, H, W]
            
            # 2.2 按该模态的系数进行权重的加权均值（每个样本的该模态奇异值）
            weights = all_singular_values + 1e-12  # [N, modes]，防止除0
            mean_weighted = np.zeros((effective_modes, H, W), dtype=np.float64)
            for m in range(effective_modes):
                w = weights[:, m]
                wsum = float(np.sum(w))
                if wsum <= 0:
                    mean_weighted[m] = mean_unweighted[m]
                else:
                    acc = np.tensordot(w, mode_matrices[m], axes=(0, 0))  # [H, W]
                    mean_weighted[m] = acc / wsum
            
            # 可视化前6个模态（非加权均值）
            n_display = min(6, effective_modes)
            cols = 3
            rows = (n_display + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
            axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]
            for i in range(rows * cols):
                ax = axes[i]
                ax.axis('off')
                if i < n_display:
                    avg_mode = mean_unweighted[i]
                    vmax = np.max(np.abs(avg_mode)) + 1e-12
                    vmin = -vmax
                    im = ax.imshow(avg_mode, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                    ax.set_title(f'模态 {i+1} 均值(未加权)')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass
            plt.tight_layout()
            mode_mean_unw_png = self.output_dir / 'per_sample_svd_mode_mean_unweighted.png'
            plt.savefig(mode_mean_unw_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                mode_mean_unw_svg = self.output_dir / 'per_sample_svd_mode_mean_unweighted.svg'
                plt.savefig(mode_mean_unw_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                mode_mean_unw_svg = mode_mean_unw_png
            plt.close(fig)
            
            # 可视化前6个模态（按系数加权均值）
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
            axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]
            for i in range(rows * cols):
                ax = axes[i]
                ax.axis('off')
                if i < n_display:
                    avg_mode = mean_weighted[i]
                    vmax = np.max(np.abs(avg_mode)) + 1e-12
                    vmin = -vmax
                    im = ax.imshow(avg_mode, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                    ax.set_title(f'模态 {i+1} 均值(按系数加权)')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass
            plt.tight_layout()
            mode_mean_w_png = self.output_dir / 'per_sample_svd_mode_mean_weighted.png'
            plt.savefig(mode_mean_w_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                mode_mean_w_svg = self.output_dir / 'per_sample_svd_mode_mean_weighted.svg'
                plt.savefig(mode_mean_w_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                mode_mean_w_svg = mode_mean_w_png
            plt.close(fig)
            
            output_paths['mode_mean_unweighted_png'] = str(mode_mean_unw_png)
            output_paths['mode_mean_unweighted_svg'] = str(mode_mean_unw_svg)
            output_paths['mode_mean_weighted_png'] = str(mode_mean_w_png)
            output_paths['mode_mean_weighted_svg'] = str(mode_mean_w_svg)
            logger.info(f"模态均值矩阵图已保存到: {mode_mean_unw_png} 与 {mode_mean_w_png}")
            
            # 保存矩阵统计数据
            mode_stats_npz = self.output_dir / 'per_sample_svd_mode_matrix_stats.npz'
            np.savez(mode_stats_npz,
                    mean_unweighted=mean_unweighted,
                    mean_weighted=mean_weighted,
                    std_unweighted=std_unweighted,
                    coeff_means=coeff_means,
                    coeff_stds=coeff_stds)
            output_paths['mode_matrix_stats_npz'] = str(mode_stats_npz)
            
        except Exception as e:
            logger.warning(f"生成模态平均矩阵失败: {e}")
        
        # 3. 加权均态重构（所有样本系数乘以样本均值的叠加）
        try:
            # 计算加权叠加：每个样本的系数 × 该样本的原始矩阵，然后在所有样本上平均
            weighted_reconstruction = np.zeros((H, W), dtype=np.float64)
            total_weight = 0.0
            
            # 方法1: 使用样本原始数据与系数的直接加权
            sample_weights = np.mean(all_singular_values, axis=1)  # 每个样本的平均系数作为权重
            for i in range(N):
                weight = sample_weights[i]
                if weight > 0:
                    weighted_reconstruction += weight * reshaped_data[i]
                    total_weight += weight
            
            if total_weight > 0:
                weighted_reconstruction /= total_weight
                
            # 方法2: 使用模态重构的加权叠加
            modal_weighted_reconstruction = np.zeros((H, W), dtype=np.float64)
            for i in range(N):
                sample_recon = np.zeros((H, W), dtype=np.float64)
                for m in range(effective_modes):
                    coeff = all_singular_values[i, m]
                    ui = all_U[i, :, m]  # [H]
                    vi = all_Vt[i, m, :]  # [W]
                    sample_recon += coeff * np.outer(ui, vi)
                modal_weighted_reconstruction += sample_recon
            modal_weighted_reconstruction /= N
            
            # 可视化对比
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 原始数据均值
            data_mean = np.mean(reshaped_data, axis=0)
            vmax_data = np.max(np.abs(data_mean)) + 1e-12
            vmin_data = -vmax_data
            im1 = axes[0,0].imshow(data_mean, cmap='RdBu_r', vmin=vmin_data, vmax=vmax_data)
            axes[0,0].set_title('原始数据均值')
            axes[0,0].axis('off')
            plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
            
            # 系数加权均值
            vmax_weighted = np.max(np.abs(weighted_reconstruction)) + 1e-12
            vmin_weighted = -vmax_weighted
            im2 = axes[0,1].imshow(weighted_reconstruction, cmap='RdBu_r', vmin=vmin_weighted, vmax=vmax_weighted)
            axes[0,1].set_title('系数加权均态')
            axes[0,1].axis('off')
            plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
            
            # 模态重构均值
            vmax_modal = np.max(np.abs(modal_weighted_reconstruction)) + 1e-12
            vmin_modal = -vmax_modal
            im3 = axes[1,0].imshow(modal_weighted_reconstruction, cmap='RdBu_r', vmin=vmin_modal, vmax=vmax_modal)
            axes[1,0].set_title('模态重构均态')
            axes[1,0].axis('off')
            plt.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)
            
            # 差异图
            diff = weighted_reconstruction - data_mean
            vmax_diff = np.max(np.abs(diff)) + 1e-12
            vmin_diff = -vmax_diff
            im4 = axes[1,1].imshow(diff, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff)
            axes[1,1].set_title('加权均态 - 原始均值')
            axes[1,1].axis('off')
            plt.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)
            
            # 字体设置
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass
            
            plt.tight_layout()
            weighted_recon_png = self.output_dir / 'per_sample_svd_weighted_reconstruction.png'
            plt.savefig(weighted_recon_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                weighted_recon_svg = self.output_dir / 'per_sample_svd_weighted_reconstruction.svg'
                plt.savefig(weighted_recon_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                weighted_recon_svg = weighted_recon_png
            plt.close(fig)
            
            output_paths['weighted_reconstruction_png'] = str(weighted_recon_png)
            output_paths['weighted_reconstruction_svg'] = str(weighted_recon_svg)
            logger.info(f"加权重构对比图已保存到: {weighted_recon_png}")
            
            # 保存重构数据
            weighted_recon_npz = self.output_dir / 'per_sample_svd_weighted_reconstruction.npz'
            np.savez(weighted_recon_npz,
                    data_mean=data_mean,
                    weighted_reconstruction=weighted_reconstruction,
                    modal_weighted_reconstruction=modal_weighted_reconstruction,
                    sample_weights=sample_weights,
                    rmse_weighted=np.sqrt(np.mean((weighted_reconstruction - data_mean)**2)),
                    rmse_modal=np.sqrt(np.mean((modal_weighted_reconstruction - data_mean)**2)))
            output_paths['weighted_reconstruction_npz'] = str(weighted_recon_npz)
            
        except Exception as e:
            logger.warning(f"生成加权重构分析失败: {e}")
        
        logger.info(f"逐样本SVD系数与矩阵分析完成，共生成 {len(output_paths)} 个文件")
        return output_paths
    
    def visualize_per_sample_leading_modes(self,
                                           per_sample_data: Dict,
                                           sample_count: int = 50,
                                           modes: Tuple[int, int] = (0, 1),
                                           scale_by_singular_value: bool = True,
                                           random: bool = False,
                                           seed: int = 42) -> Dict[str, str]:
        """可视化逐样本的第一/第二模态，按样本直接展示（不做均值化），统一colorbar。
        - 将选取的样本按列横向排布，行表示不同模态（默认两行：模态1与模态2）。
        - 每个子图展示该样本该模态的矩阵：
            如果 scale_by_singular_value=True，则显示 sigma_i,m * (U_i[:,m] ⊗ V_i[m,:])；
            否则显示 U_i[:,m] ⊗ V_i[m,:]。
        - colorbar 统一到整张大图（跨两行所有子图）。
        """
        try:
            all_U = per_sample_data['all_U']            # [N, H, k]
            all_Vt = per_sample_data['all_Vt']          # [N, k, W]
            all_singular_values = per_sample_data['all_singular_values']  # [N, k]
            H, W = per_sample_data['spatial_shape']
            N, k = all_singular_values.shape
        except Exception as e:
            logger.warning(f"逐样本领先模态可视化输入不完整: {e}")
            return {}

        # 校验模态索引
        max_mode_index = max(modes)
        if k <= max_mode_index:
            logger.warning(f"有效模态数为{k}，不足以显示请求的模态索引 {modes}")
            return {}

        # 抽样索引（默认等距抽样，以覆盖全体样本；如需随机可开启 random）
        if N <= sample_count:
            sample_indices = np.arange(N)
        else:
            if random:
                rng = np.random.default_rng(seed)
                sample_indices = np.sort(rng.choice(N, size=sample_count, replace=False))
            else:
                sample_indices = np.linspace(0, N - 1, sample_count, dtype=int)
        sample_indices = np.asarray(sample_indices, dtype=int)
        C = len(sample_indices)

        # 准备所有面板的数据并计算全局colorbar范围（对两行所有子图统一）
        panels = []  # list of (mode_idx, sample_idx, panel_matrix)
        global_abs_max = 0.0
        for m in modes:
            for i in sample_indices:
                ui = all_U[i, :, m]      # [H]
                vi = all_Vt[i, m, :]     # [W]
                mat = np.outer(ui, vi)   # [H, W]
                if scale_by_singular_value:
                    s = all_singular_values[i, m]
                    mat = s * mat
                panels.append((m, int(i), mat))
                amax = float(np.max(np.abs(mat))) if mat.size > 0 else 0.0
                if amax > global_abs_max:
                    global_abs_max = amax
        global_abs_max = global_abs_max + 1e-12
        vmin, vmax = -global_abs_max, global_abs_max

        # 组织子图：行=模态数，列=样本数，使用 constrained_layout
        rows = len(modes)
        cols = C
        figsize = (max(10.0, cols * 1.1), max(3.0, rows * 1.6))
        fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if cols == 1:
            axes = np.expand_dims(axes, axis=1)

        # 绘制，整图使用单一 colorbar，范围取该图数据自身范围，避免遮挡
        cmap = 'RdBu_r'
        im_for_cbar = None
        idx = 0
        for r, m in enumerate(modes):
            for c, i in enumerate(sample_indices):
                ax = axes[r, c]
                ax.axis('off')
                _, _, mat = panels[idx]
                im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
                if im_for_cbar is None:
                    im_for_cbar = im
                idx += 1

        # 整图 colorbar（使用 constrained_layout，设置合适的 pad/fraction 防止遮挡）
        try:
            cbar = fig.colorbar(im_for_cbar, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
            fam = CHINESE_FONT_NAME
            if fam:
                cbar.ax.tick_params(labelsize=8)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontfamily(fam)
        except Exception as e:
            logger.warning(f"添加统一colorbar失败: {e}")

        # 标题
        try:
            import matplotlib as mpl
            fam = CHINESE_FONT_NAME
            title = f"逐样本领先模态对比（未均值化，样本数={C}，模态={','.join([str(m+1) for m in modes])}）"
            if fam:
                fig.suptitle(title, fontfamily=fam, fontsize=12)
            else:
                fig.suptitle(title, fontsize=12)
        except Exception:
            pass

        # 使用 constrained_layout 已经启用自动布局，无需再调用 tight_layout

        # 保存
        out_png = self.output_dir / f'per_sample_leading_modes_m{modes[0]+1}_m{modes[1]+1}_samples{C}.png'
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        try:
            import matplotlib as mpl
            old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
            mpl.rcParams['svg.fonttype'] = 'none'
            out_svg = self.output_dir / f'per_sample_leading_modes_m{modes[0]+1}_m{modes[1]+1}_samples{C}.svg'
            plt.savefig(out_svg, format='svg', bbox_inches='tight')
            if old_svg_fonttype is not None:
                mpl.rcParams['svg.fonttype'] = old_svg_fonttype
        except Exception:
            out_svg = out_png
        plt.close(fig)

        # 保存索引
        idx_path = self.output_dir / f'per_sample_leading_modes_indices.npy'
        try:
            np.save(idx_path, sample_indices)
        except Exception:
            idx_path = None

        logger.info(f"逐样本领先模态可视化已保存到: {out_png}")
        return {
            'leading_modes_png': str(out_png),
            'leading_modes_svg': str(out_svg),
            'leading_modes_indices_npy': str(idx_path) if idx_path is not None else ''
        }
    
    def visualize_per_sample_modes_separate(self,
                                            per_sample_data: Dict,
                                            sample_count: int = 50,
                                            modes: Optional[List[int]] = None,
                                            scale_by_singular_value: bool = True,
                                            random: bool = False,
                                            seed: int = 42,
                                            max_cols: int = 10) -> Dict[str, Dict[str, str]]:
        """为每个模态分别生成一张大图，文件后缀 m1, m2, m3 ...，并按行列自适应铺排（非单行）。
        - colorbar 统一到整张图（每个模态一张图）。
        - 不做均值化，直接使用每个样本自己的 U、V、sigma。
        """
        try:
            all_U = per_sample_data['all_U']            # [N, H, k]
            all_Vt = per_sample_data['all_Vt']          # [N, k, W]
            all_singular_values = per_sample_data['all_singular_values']  # [N, k]
            H, W = per_sample_data['spatial_shape']
            N, k = all_singular_values.shape
        except Exception as e:
            logger.warning(f"逐样本分模态可视化输入不完整: {e}")
            return {}

        # 模态集合
        if modes is None:
            modes = list(range(min(k, 8)))  # 默认最多前8个模态，避免生成过多大图
        else:
            # 过滤非法索引
            modes = [m for m in modes if 0 <= m < k]
        if len(modes) == 0:
            logger.warning("无可用模态索引用于可视化")
            return {}

        # 抽样索引（默认等距抽样）
        if N <= sample_count:
            sample_indices = np.arange(N)
        else:
            if random:
                rng = np.random.default_rng(seed)
                sample_indices = np.sort(rng.choice(N, size=sample_count, replace=False))
            else:
                sample_indices = np.linspace(0, N - 1, sample_count, dtype=int)
        sample_indices = np.asarray(sample_indices, dtype=int)
        C = len(sample_indices)

        # 计算自适应行列数：列不超过 max_cols，行据此确定
        cols = min(max_cols, C)
        rows = int(np.ceil(C / cols))

        results: Dict[str, Dict[str, str]] = {}
        cmap = 'RdBu_r'

        for m in modes:
            # 预计算全局colorbar范围
            global_abs_max = 0.0
            mats = []
            for i in sample_indices:
                ui = all_U[i, :, m]
                vi = all_Vt[i, m, :]
                mat = np.outer(ui, vi)
                if scale_by_singular_value:
                    s = all_singular_values[i, m]
                    mat = s * mat
                mats.append(mat)
                amax = float(np.max(np.abs(mat))) if mat.size > 0 else 0.0
                if amax > global_abs_max:
                    global_abs_max = amax
            global_abs_max = global_abs_max + 1e-12
            vmin, vmax = -global_abs_max, global_abs_max

            # 画布与子图，使用 constrained_layout
            figsize = (max(8.0, cols * 1.2), max(6.0, rows * 1.2))
            fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = np.expand_dims(axes, axis=0)
            elif cols == 1:
                axes = np.expand_dims(axes, axis=1)

            # 绘制：整图单一 colorbar，范围取该图(该模态)所有子图的共同范围
            im_for_cbar = None
            for idx, i in enumerate(sample_indices):
                r = idx // cols
                c = idx % cols
                ax = axes[r, c]
                ax.axis('off')
                mat = mats[idx]
                im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
                if im_for_cbar is None:
                    im_for_cbar = im
            # 对于多余的空格子，关掉坐标轴
            total_cells = rows * cols
            for idx in range(C, total_cells):
                r = idx // cols
                c = idx % cols
                axes[r, c].axis('off')

            # 整图 colorbar
            try:
                cbar = fig.colorbar(im_for_cbar, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
                fam = CHINESE_FONT_NAME
                if fam:
                    cbar.ax.tick_params(labelsize=8)
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontfamily(fam)
            except Exception as e:
                logger.warning(f"添加统一colorbar失败(m={m}): {e}")

            # 标题
            try:
                fam = CHINESE_FONT_NAME
                title = f"逐样本模态 {m+1}（未均值化，样本数={C}）"
                if fam:
                    fig.suptitle(title, fontfamily=fam, fontsize=12)
                else:
                    fig.suptitle(title, fontsize=12)
            except Exception:
                pass

            # 使用 constrained_layout 已经启用自动布局，无需再调用 tight_layout

            # 保存文件：m{m+1}
            out_png = self.output_dir / f'per_sample_modes_m{m+1}_samples{C}.png'
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                out_svg = self.output_dir / f'per_sample_modes_m{m+1}_samples{C}.svg'
                plt.savefig(out_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                out_svg = out_png
            plt.close(fig)

            results[f'm{m+1}'] = {
                'png': str(out_png),
                'svg': str(out_svg)
            }

        # 保存抽样索引（共用一份）
        idx_path = self.output_dir / f'per_sample_modes_indices.npy'
        try:
            np.save(idx_path, sample_indices)
            for kkey in results:
                results[kkey]['indices_npy'] = str(idx_path)
        except Exception:
            pass

        logger.info(f"分模态逐样本可视化完成，共生成 {len(results)} 张图")
        return results
    
    def save_analysis_report(self, analysis_results: Dict, output_dir: Path):
        """生成分析报告并保存到文件"""
        summary_path = output_dir / 'output_modality_summary.md'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 输出模态分析总结\n\n")
            
            # 基本信息
            f.write("## 基本信息\n")
            f.write(f"- 输出维度: {analysis_results['output_shape']}\n")
            f.write(f"- 样本总数: {analysis_results['n_samples']}\n")
            f.write(f"- 分析时间: {analysis_results['timestamp']}\n")
            f.write(f"- 保留维度(有效模态): {analysis_results.get('effective_modes', 'N/A')}\n")
            f.write(f"- 降维前形状: {analysis_results['original_shape']}\n")
            f.write(f"- 降维后形状: {analysis_results['reduced_shape']}\n\n")
            
            # 流程说明
            f.write("## 流程说明\n")
            f.write("本分析工具对 HDF5 数据文件进行输出模态分析，流程如下：\n\n")
            f.write("### 1. 数据定位与加载\n")
            f.write("- 定位 HDF5 文件中的 `tensor` 数据集\n")
            f.write("- 加载数据并记录原始形状信息\n\n")
            f.write("### 2. 预处理\n")
            f.write("- 4D 数据（[样本, 时间, 高, 宽]）：选择最后一个时间步作为输出\n")
            f.write("- 3D 数据（[样本, 高, 宽]）：直接使用\n")
            f.write("- 随机抽样（如果样本数超过最大值）\n")
            f.write("- 空间维度展平：[样本数, H×W]\n\n")
            
            cfg = analysis_results.get('config', {})
            method = cfg.get('method', 'standard')
            
            if method == 'per_sample_averaged':
                f.write("### 3. 逐样本 SVD 分析\n")
                f.write("- 对每个样本的 H×W 矩阵进行独立 SVD 分解\n")
                f.write("- 计算每个样本的右奇异向量 V^T\n")
                f.write("- 对所有样本的 V^T 进行平均，得到统一的空间模态基\n")
                f.write("- 重新投影所有样本到平均模态基上\n")
                f.write("- 根据能量阈值自动确定有效模态数\n\n")
            else:
                f.write("### 3. 标准 SVD 分析\n")
                f.write("- 数据标准化（零均值，单位方差）\n")
                f.write("- 对整体数据矩阵进行 SVD 分解：Data = U × S × V^T\n")
                f.write("- 根据累积能量和能量阈值确定有效模态数\n")
                f.write("- 提取降维后的样本嵌入（U 矩阵的前若干列）\n\n")
            
            f.write("### 4. 模态系数统计与可视化\n")
            if 'coefficients_visualizations' in analysis_results:
                f.write("- 生成系数均值柱状图：显示每个模态的平均激活强度\n")
                f.write("- 绘制系数分布箱线图：展示每个模态系数的分布特征\n")
                f.write("- 计算系数相关性热力图：分析不同模态间的相关性\n\n")
            else:
                f.write("- （本次运行未启用系数可视化）\n\n")
            
            f.write("### 5. 样本重构与相似度分析\n")
            if 'similarity' in analysis_results:
                f.write("- 基于样本嵌入计算余弦相似度和欧氏距离\n")
                f.write("- 生成样本间相似度热力图（余弦相似度与欧氏距离对比）\n")
                f.write("- 统计相似度分布（均值、标准差、最值）\n\n")
            else:
                f.write("- （本次运行未启用样本相似度分析）\n\n")
            
            if 'per_mode_similarity' in analysis_results:
                f.write("### 6. 每模态样本相似度分析\n")
                f.write("- 针对每个模态维度单独计算样本间相似度\n")
                f.write("- 为每个模态生成独立的相似度热力图\n")
                f.write("- 分析不同模态维度下的样本聚类特征\n\n")
            
            if 'reconstruction_similarity' in analysis_results:
                f.write("### 7. 重构相似度分析\n")
                f.write("- 针对不同的模态保留数量 k，评估重构质量\n")
                f.write("- 计算样本对间的重构相似度和相对误差\n")
                f.write("- 生成重构质量随模态数变化的可视化图表\n\n")
            
            f.write("### 8. 可视化与颜色映射规范\n")
            f.write("- **余弦相似度**：使用 RdYlBu_r 配色，范围 [-1, 1]，中心值 0\n")
            f.write("- **欧氏距离**：使用 viridis 配色，范围 [0, max_distance]\n")
            f.write("- **相关性热力图**：使用 RdBu_r 配色，范围 [-1, 1]，中心值 0\n")
            f.write("- 所有图表均提供 PNG（高分辨率）和 SVG（矢量）两种格式\n\n")
            
            f.write("### 9. 输出结构说明\n")
            f.write("- **可视化文件**：模态基可视化、系数分析图表、相似度热力图等\n")
            f.write("- **数值文件**：相似度矩阵（.npy）、重构统计（.npz）等\n")
            f.write("- **分析报告**：本 Markdown 文件，包含完整的分析结果摘要\n\n")
            
            # 运行参数记录
            f.write("### 本次运行参数\n")
            f.write(f"- 最大样本数: {analysis_results.get('n_samples', 'N/A')}\n")
            f.write(f"- 模态数设置: {cfg.get('n_modes', 'N/A')}\n")
            f.write(f"- 能量阈值: {cfg.get('energy_threshold', 'N/A'):.1%}\n")
            f.write(f"- 分析方法: {method}\n")
            if 'similarity' in analysis_results:
                f.write("- 启用样本相似度分析\n")
            if 'coefficients_visualizations' in analysis_results:
                f.write("- 启用系数可视化\n")
            if 'per_mode_similarity' in analysis_results:
                f.write("- 启用每模态相似度分析\n")
            if 'reconstruction_similarity' in analysis_results:
                f.write("- 启用重构相似度分析\n")
            f.write("\n")
            
            # SVD分析结果
            f.write("## SVD分析结果\n")
            S = analysis_results.get('singular_values', None)
            cfg = analysis_results.get('config', {})
            eff = analysis_results.get('effective_modes', 0)
            
            # 检查是否为逐样本SVD方法
            method = cfg.get('method', 'standard')
            if method == 'per_sample_averaged':
                f.write("- **分析方法**: 逐样本SVD（对每个样本的H×W矩阵单独进行SVD，然后对右奇异向量进行平均）\n")
            else:
                f.write("- **分析方法**: 标准SVD（对整体数据矩阵进行SVD分解）\n")
            
            if S is not None:
                total_energy = float(np.sum(S**2))
                retained_energy = float(np.sum(S[:eff]**2)) if eff > 0 else 0.0
                retention = (retained_energy / total_energy) if total_energy > 0 else 0.0
                f.write(f"- 模态总数: {len(S)}\n")
                f.write(f"- 能量阈值: {cfg.get('energy_threshold', 0.0):.1%}\n")
                f.write(f"- 保留奇异值数(有效模态): {eff}\n")
                f.write(f"- 总能量: {total_energy:.6f}\n")
                f.write(f"- 保留能量: {retained_energy:.6f}\n")
                f.write(f"- 能量保留率: {retention:.1%}\n")
                
                # 如果是逐样本SVD，添加额外统计信息
                if method == 'per_sample_averaged':
                    per_sample_stats = analysis_results.get('per_sample_stats', {})
                    if per_sample_stats:
                        f.write(f"- **逐样本统计**: 每个样本独立SVD后平均得到的空间模态基\n")
                        avg_energy = per_sample_stats.get('avg_energy_per_mode', [])
                        if avg_energy and len(avg_energy) > 0:
                            f.write(f"- 平均能量比例（前5模态）: {[f'{e:.3f}' for e in avg_energy[:5]]}\n")
                f.write("\n")
            else:
                f.write("- 无奇异值信息\n\n")
            
            
            # 可视化文件路径
            f.write("## 可视化文件\n")
            for key, value in analysis_results.get('visualizations', {}).items():
                if isinstance(value, Path):
                    f.write(f"- {key}: {value.name}\n")
                else:
                    f.write(f"- {key}: {value}\n")
            
            # 每模态样本相似度热力图（如果有）
            if 'per_mode_similarity' in analysis_results:
                f.write("\n## 每模态样本相似度热力图\n")
                per_mode = analysis_results['per_mode_similarity']
                heatmaps = per_mode.get('heatmaps', {})
                for k, v in heatmaps.items():
                    f.write(f"- {k}: {v}\n")
                summary = per_mode.get('summary', {})
                if summary:
                    f.write(f"- 概要: {summary}\n")
            
            # 新增：系数可视化路径（如果有）
            if 'coefficients_visualizations' in analysis_results:
                f.write("\n## 系数可视化\n")
                coeff_viz = analysis_results['coefficients_visualizations']
                # 兼容字段名称
                if 'mean_bar_png' in coeff_viz:
                    f.write(f"- 系数均值柱状图(PNG): {coeff_viz['mean_bar_png']}\n")
                if 'mean_bar_svg' in coeff_viz:
                    f.write(f"- 系数均值柱状图(SVG): {coeff_viz['mean_bar_svg']}\n")
                if 'boxplot_png' in coeff_viz:
                    f.write(f"- 系数分布箱线图(PNG): {coeff_viz['boxplot_png']}\n")
                if 'boxplot_svg' in coeff_viz:
                    f.write(f"- 系数分布箱线图(SVG): {coeff_viz['boxplot_svg']}\n")
                if 'correlation_png' in coeff_viz:
                    f.write(f"- 系数相关性热力图(PNG): {coeff_viz['correlation_png']}\n")
                if 'correlation_svg' in coeff_viz:
                    f.write(f"- 系数相关性热力图(SVG): {coeff_viz['correlation_svg']}\n")

            # 新增：逐样本SVD系数与矩阵分析输出（如果有）
            if 'per_sample_svd_coefficients' in analysis_results:
                f.write("\n## 逐样本SVD系数与矩阵分析输出\n")
                p = analysis_results['per_sample_svd_coefficients']
                if 'coefficients_analysis_png' in p:
                    f.write(f"- 系数分析汇总图(PNG): {p['coefficients_analysis_png']}\n")
                if 'coefficients_analysis_svg' in p:
                    f.write(f"- 系数分析汇总图(SVG): {p['coefficients_analysis_svg']}\n")
                if 'mode_mean_unweighted_png' in p:
                    f.write(f"- 非加权模态均值(PNG): {p['mode_mean_unweighted_png']}\n")
                if 'mode_mean_unweighted_svg' in p:
                    f.write(f"- 非加权模态均值(SVG): {p['mode_mean_unweighted_svg']}\n")
                if 'mode_mean_weighted_png' in p:
                    f.write(f"- 加权模态均值(PNG): {p['mode_mean_weighted_png']}\n")
                if 'mode_mean_weighted_svg' in p:
                    f.write(f"- 加权模态均值(SVG): {p['mode_mean_weighted_svg']}\n")
                if 'mode_matrix_stats_npz' in p:
                    f.write(f"- 模态矩阵统计(NPZ): {p['mode_matrix_stats_npz']}\n")
                if 'weighted_reconstruction_png' in p:
                    f.write(f"- 加权重构对比(PNG): {p['weighted_reconstruction_png']}\n")
                if 'weighted_reconstruction_svg' in p:
                    f.write(f"- 加权重构对比(SVG): {p['weighted_reconstruction_svg']}\n")
                if 'weighted_reconstruction_npz' in p:
                    f.write(f"- 加权重构数据(NPZ): {p['weighted_reconstruction_npz']}\n")
            # 新增：逐样本领先模态横向大图（如果有）
            if 'per_sample_leading_modes' in analysis_results:
                lm = analysis_results['per_sample_leading_modes']
                f.write("\n## 逐样本领先模态（未均值化）\n")
                if 'leading_modes_png' in lm:
                    f.write(f"- 横向对比大图(PNG): {lm['leading_modes_png']}\n")
                if 'leading_modes_svg' in lm:
                    f.write(f"- 横向对比大图(SVG): {lm['leading_modes_svg']}\n")
                if 'leading_modes_indices_npy' in lm and lm['leading_modes_indices_npy']:
                    f.write(f"- 抽样样本索引(NPY): {lm['leading_modes_indices_npy']}\n")

            # 新增：逐样本分模态大图（如果有）
            if 'per_sample_modes_separate' in analysis_results:
                sep = analysis_results['per_sample_modes_separate']
                f.write("\n## 逐样本分模态可视化（未均值化，自适应行列）\n")
                for mk in sorted(sep.keys()):
                    item = sep[mk]
                    f.write(f"- 模态 {mk.upper()} 图(PNG): {item.get('png','')}\n")
                    f.write(f"  模态 {mk.upper()} 图(SVG): {item.get('svg','')}\n")
                any_item = next(iter(sep.values())) if len(sep)>0 else None
                if any_item and 'indices_npy' in any_item:
                    f.write(f"- 抽样样本索引(NPY): {any_item['indices_npy']}\n")

            # 相似度分析（如果有）
            if 'similarity' in analysis_results:
                f.write("\n## 样本相似度分析\n")
                sim = analysis_results['similarity']
                summary = sim['summary']
                f.write(f"- 平均相似度: {summary['avg']:.3f}\n")
                f.write(f"- 相似度标准差: {summary['std']:.3f}\n")
                f.write(f"- 最高相似度: {summary['max']:.3f}\n")
                f.write(f"- 最低相似度: {summary['min']:.3f}\n")
                # 兼容旧字段：余弦热力图
                f.write(f"- 余弦相似度热力图: {sim.get('heatmap_png', 'N/A')}\n")
                # 新增：欧氏距离与组合图路径
                hmaps = sim.get('heatmaps', {})
                if hmaps:
                    if 'euclidean_png' in hmaps:
                        f.write(f"- 欧氏距离热力图: {hmaps.get('euclidean_png')}\n")
                    if 'combined_png' in hmaps:
                        f.write(f"- 相似度组合图(余弦+欧氏): {hmaps.get('combined_png')}\n")
            # 新增：2D模态统计
            # 新增：在报告中加入 2D 模态均值柱状展示路径
            if 'mode_2d_stats' in analysis_results:
                f.write("\n## 2D 模态系数与矩阵分布\n")
                m2d = analysis_results['mode_2d_stats']
                # HxW 模态矩阵展示（来自可视化结果）
                viz = analysis_results.get('visualizations', {})
                if isinstance(viz, dict):
                    if 'grid_png' in viz and viz['grid_png']:
                        f.write(f"- 模态矩阵(HxW)网格汇总: {viz['grid_png']}\n")
                    if 'single_png' in viz and isinstance(viz['single_png'], list) and len(viz['single_png']) > 0:
                        sample_list = viz['single_png'][:min(9, len(viz['single_png']))]
                        f.write(f"- 单模态(HxW)PNG共 {len(viz['single_png'])} 张，示例: {sample_list}\n")
                if 'coeff_dist_png' in m2d:
                    f.write(f"- 系数分布直方图: {m2d['coeff_dist_png']}\n")
                if 'spatial_dist_png' in m2d:
                    f.write(f"- 重构样本空间分布(均值/标准差): {m2d['spatial_dist_png']}\n")
                if 'coeff_matrix_png' in m2d:
                    f.write(f"- 系数矩阵热力图: {m2d['coeff_matrix_png']}\n")
                if 'mode_matrix_hist_png' in m2d:
                    f.write(f"- 模态矩阵值分布直方图: {m2d['mode_matrix_hist_png']}\n")
                if 'mode_dist_3d_png' in m2d:
                    f.write(f"- 模态空间分布3D(前若干模态): {m2d['mode_dist_3d_png']}\n")
                if 'coeff_mean_png' in m2d:
                    f.write(f"- 系数均值柱状图: {m2d['coeff_mean_png']}\n")
                if 'mode_mean_png' in m2d:
                    f.write(f"- 模态矩阵均值热力图: {m2d['mode_mean_png']}\n")
                if 'mode_mean_bar_png' in m2d:
                    f.write(f"- 模态矩阵均值柱状(2D): {m2d['mode_mean_bar_png']}\n")
                if 'superposition_png' in m2d:
                    f.write(f"- 均值叠加态热力图: {m2d['superposition_png']}\n")

        logger.info(f"分析报告已保存到: {summary_path}")
        return summary_path

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray, data_range: float = None) -> float:
        """计算结构相似性指数(SSIM)，简化版实现"""
        try:
            if data_range is None:
                data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
                
            if data_range == 0:
                return 1.0 if np.allclose(img1, img2) else 0.0
            
            # 计算均值
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            
            # 计算方差和协方差
            var1 = np.var(img1)
            var2 = np.var(img2)
            cov12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            # SSIM常数
            c1 = (0.01 * data_range) ** 2
            c2 = (0.03 * data_range) ** 2
            
            # SSIM公式
            ssim = ((2 * mu1 * mu2 + c1) * (2 * cov12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
                   
            return float(ssim)
        except Exception:
            return 0.0

    def _compute_psnr(self, img1: np.ndarray, img2: np.ndarray, data_range: float = None) -> float:
        """计算峰值信噪比(PSNR)"""
        try:
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
                
            if data_range is None:
                data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
                
            if data_range == 0:
                return float('inf')
                
            psnr = 20 * np.log10(data_range / np.sqrt(mse))
            return float(psnr)
        except Exception:
            return 0.0

    def analyze_sample_reconstruction_similarity(
        self, 
        original_data: np.ndarray, 
        Vt_reduced: np.ndarray, 
        coefficients: np.ndarray, 
        spatial_shape: Tuple[int, int],
        k_range: List[int] = None,
        max_samples_to_analyze: int = 50
    ) -> Dict[str, str]:
        """
        分析每个样本在不同特征数k下的重构相似度
        
        Args:
            original_data: 原始数据 [n_samples, H*W]
            Vt_reduced: SVD右奇异向量 [k_max, H*W]
            coefficients: 投影系数 [n_samples, k_max]
            spatial_shape: 空间形状 (H, W)
            k_range: 要测试的k值列表，默认为[1,2,4,8,16,32,64]
            max_samples_to_analyze: 最大分析样本数
            
        Returns:
            保存的文件路径字典
        """
        logger.info("开始样本重构相似度分析...")
        
        H, W = spatial_shape
        n_samples_total, feature_dim = original_data.shape
        k_max = Vt_reduced.shape[0]
        
        # 设置默认k范围
        if k_range is None:
            k_range = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= k_max]
        else:
            k_range = [k for k in k_range if k <= k_max]
            
        logger.info(f"分析k值范围: {k_range}")
        
        # 限制分析样本数
        n_samples = min(n_samples_total, max_samples_to_analyze)
        sample_indices = np.random.choice(n_samples_total, n_samples, replace=False) if n_samples_total > n_samples else np.arange(n_samples_total)
        
        # 准备结果存储
        similarity_results = {
            'cosine': np.zeros((n_samples, len(k_range))),
            'pearson': np.zeros((n_samples, len(k_range))),
            'ssim': np.zeros((n_samples, len(k_range))),
            'psnr': np.zeros((n_samples, len(k_range))),
            'mse': np.zeros((n_samples, len(k_range))),
            'relative_error': np.zeros((n_samples, len(k_range)))
        }
        
        # 计算数据范围用于SSIM和PSNR
        data_range = original_data.max() - original_data.min()
        
        logger.info(f"开始计算 {n_samples} 个样本在 {len(k_range)} 个k值下的相似度...")
        
        for i, sample_idx in enumerate(sample_indices):
            original_sample = original_data[sample_idx]  # [H*W]
            original_2d = original_sample.reshape(H, W)  # [H, W]
            
            for j, k in enumerate(k_range):
                # 重构样本：使用前k个模态
                coeff_k = coefficients[sample_idx, :k]  # [k]
                Vt_k = Vt_reduced[:k, :]  # [k, H*W]
                reconstructed_sample = coeff_k @ Vt_k  # [H*W]
                reconstructed_2d = reconstructed_sample.reshape(H, W)  # [H, W]
                
                # 1. 余弦相似度（基于展平数据）
                norm_orig = np.linalg.norm(original_sample)
                norm_recon = np.linalg.norm(reconstructed_sample)
                if norm_orig > 0 and norm_recon > 0:
                    cosine_sim = np.dot(original_sample, reconstructed_sample) / (norm_orig * norm_recon)
                else:
                    cosine_sim = 1.0 if np.allclose(original_sample, reconstructed_sample) else 0.0
                similarity_results['cosine'][i, j] = cosine_sim
                
                # 2. 皮尔逊相关系数
                try:
                    pearson_corr = np.corrcoef(original_sample, reconstructed_sample)[0, 1]
                    if np.isnan(pearson_corr):
                        pearson_corr = 1.0 if np.allclose(original_sample, reconstructed_sample) else 0.0
                except Exception:
                    pearson_corr = 0.0
                similarity_results['pearson'][i, j] = pearson_corr
                
                # 3. SSIM（基于2D图像）
                ssim_val = self._compute_ssim(original_2d, reconstructed_2d, data_range)
                similarity_results['ssim'][i, j] = ssim_val
                
                # 4. PSNR（基于2D图像）
                psnr_val = self._compute_psnr(original_2d, reconstructed_2d, data_range)
                if psnr_val == float('inf'):
                    psnr_val = 100.0  # 设置一个合理的上限
                similarity_results['psnr'][i, j] = psnr_val
                
                # 5. MSE
                mse_val = np.mean((original_sample - reconstructed_sample) ** 2)
                similarity_results['mse'][i, j] = mse_val
                
                # 6. 相对误差
                rel_error = np.linalg.norm(original_sample - reconstructed_sample) / (np.linalg.norm(original_sample) + 1e-12)
                similarity_results['relative_error'][i, j] = rel_error
        
        # 保存数值结果
        similarity_stats_path = self.output_dir / 'sample_reconstruction_similarity_stats.npz'
        np.savez(similarity_stats_path,
                 k_range=k_range,
                 sample_indices=sample_indices,
                 spatial_shape=spatial_shape,
                 **similarity_results)
        
        output = {
            'similarity_stats_npz': str(similarity_stats_path)
        }
        
        # 生成可视化
        output.update(self._visualize_reconstruction_similarity(similarity_results, k_range, n_samples))
        
        logger.info(f"样本重构相似度分析完成，结果保存到: {similarity_stats_path}")
        return output

    def _visualize_reconstruction_similarity(self, similarity_results: Dict, k_range: List[int], n_samples: int) -> Dict[str, str]:
        """生成相似度分析的可视化结果"""
        output = {}
        
        # 1. 相似度随k变化的统计曲线图
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.ravel()
            
            metrics = ['cosine', 'pearson', 'ssim', 'psnr', 'mse', 'relative_error']
            metric_labels = ['余弦相似度', '皮尔逊相关系数', 'SSIM', 'PSNR (dB)', 'MSE', '相对误差']
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]
                data = similarity_results[metric]  # [n_samples, len(k_range)]
                
                # 计算统计量
                mean_vals = np.mean(data, axis=0)
                std_vals = np.std(data, axis=0)
                median_vals = np.median(data, axis=0)
                
                # 绘制均值曲线和标准差区域
                ax.plot(k_range, mean_vals, 'b-', linewidth=2, label='均值')
                ax.fill_between(k_range, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, label='±1σ')
                ax.plot(k_range, median_vals, 'r--', linewidth=1.5, label='中位数')
                
                ax.set_xlabel('模态数 k')
                ax.set_ylabel(label)
                ax.set_title(f'{label} vs 模态数 (n={n_samples})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log', base=2) if len(k_range) > 4 else None
            
            # 字体设置
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass
            
            plt.tight_layout()
            similarity_curves_png = self.output_dir / 'sample_similarity_curves.png'
            plt.savefig(similarity_curves_png, dpi=300, bbox_inches='tight')
            plt.close(fig)
            output['similarity_curves_png'] = str(similarity_curves_png)
            
        except Exception as e:
            logger.warning(f"生成相似度曲线图失败: {e}")
        
        # 2. 相似度热力图（样本 vs k值）
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.ravel()
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]
                data = similarity_results[metric]  # [n_samples, len(k_range)]
                
                # 创建热力图
                im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
                
                # 设置刻度
                ax.set_xticks(range(len(k_range)))
                ax.set_xticklabels([str(k) for k in k_range])
                ax.set_xlabel('模态数 k')
                ax.set_ylabel('样本索引')
                ax.set_title(f'{label} 热力图')
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(label)
            
            # 字体设置
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass
            
            plt.tight_layout()
            similarity_heatmap_png = self.output_dir / 'sample_similarity_heatmap.png'
            plt.savefig(similarity_heatmap_png, dpi=300, bbox_inches='tight')
            plt.close(fig)
            output['similarity_heatmap_png'] = str(similarity_heatmap_png)
            
        except Exception as e:
            logger.warning(f"生成相似度热力图失败: {e}")
        
        # 3. 散点图：不同指标间的关系
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            # 选择几个k值进行散点分析
            k_indices_to_plot = [0, len(k_range)//2, -1] if len(k_range) > 2 else [0, -1]
            
            scatter_pairs = [
                ('cosine', 'ssim', '余弦相似度', 'SSIM'),
                ('pearson', 'ssim', '皮尔逊相关', 'SSIM'),
                ('cosine', 'psnr', '余弦相似度', 'PSNR'),
                ('ssim', 'psnr', 'SSIM', 'PSNR')
            ]
            
            for i, (x_metric, y_metric, x_label, y_label) in enumerate(scatter_pairs):
                ax = axes[i]
                
                for k_idx in k_indices_to_plot:
                    k_val = k_range[k_idx]
                    x_data = similarity_results[x_metric][:, k_idx]
                    y_data = similarity_results[y_metric][:, k_idx]
                    
                    ax.scatter(x_data, y_data, alpha=0.6, s=30, label=f'k={k_val}')
                
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f'{x_label} vs {y_label}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 字体设置
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass
            
            plt.tight_layout()
            similarity_scatter_png = self.output_dir / 'sample_similarity_scatter.png'
            plt.savefig(similarity_scatter_png, dpi=300, bbox_inches='tight')
            plt.close(fig)
            output['similarity_scatter_png'] = str(similarity_scatter_png)
            
        except Exception as e:
            logger.warning(f"生成相似度散点图失败: {e}")
        
        return output

    def analyze_2d_modes(self, Vt_reduced: np.ndarray, coefficients: np.ndarray, spatial_shape: Tuple[int, int]) -> Dict[str, str]:
        """在保留2D空间结构下分析模态的样本分布：
        - 系数分布直方图 (展示各模态系数在所有样本中的分布)
        - 模态矩阵分布统计 (分析空间位置的分布特征)
        - 叠加态分布统计 (按系数分布重构的空间分布特征)
        并将结果进行可视化与保存。
        """
        try:
            H, W = int(spatial_shape[0]), int(spatial_shape[1])
        except Exception:
            logger.error(f"非法的空间形状: {spatial_shape}")
            return {}
        
        k = Vt_reduced.shape[0]
        n_samples = coefficients.shape[0]
        modes = Vt_reduced.reshape(k, H, W)  # [k, H, W]
        
        # 计算分布统计
        coeff_mean = np.mean(coefficients, axis=0)  # [k] 均值
        coeff_std = np.std(coefficients, axis=0)   # [k] 标准差
        coeff_median = np.median(coefficients, axis=0)  # [k] 中位数
        
        # 计算叠加态的分布（每个样本的重构结果）
        reconstructed_samples = []
        for i in range(min(n_samples, 100)):  # 限制样本数以节省内存
            sample_reconstruction = np.tensordot(coefficients[i], modes, axes=(0, 0))  # [H, W]
            reconstructed_samples.append(sample_reconstruction)
        reconstructed_samples = np.array(reconstructed_samples)  # [n_samples, H, W]
        
        # 保存数值结果
        coeff_stats_path = self.output_dir / 'coeff_distribution_stats.npz'
        recon_stats_path = self.output_dir / 'reconstruction_distribution_stats.npz'
        
        np.savez(coeff_stats_path, 
                mean=coeff_mean, 
                std=coeff_std, 
                median=coeff_median,
                raw_coefficients=coefficients)
        
        recon_mean = np.mean(reconstructed_samples, axis=0)  # [H, W]
        recon_std = np.std(reconstructed_samples, axis=0)    # [H, W]
        np.savez(recon_stats_path,
                mean=recon_mean,
                std=recon_std,
                samples=reconstructed_samples[:10])  # 保存前10个样本示例
        
        output: Dict[str, str] = {
            'coeff_stats_npz': str(coeff_stats_path),
            'recon_stats_npz': str(recon_stats_path),
        }
        
        # 1) 系数分布直方图 (多子图展示各模态的系数分布)
        try:
            # 决定子图布局
            cols = min(4, k)
            rows = (k + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2.5))
            if k == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if isinstance(axes, plt.Axes) else axes
            else:
                axes = axes.ravel()
            
            for i in range(k):
                ax = axes[i] if i < len(axes) else None
                if ax is None:
                    continue
                    
                # 绘制直方图
                ax.hist(coefficients[:, i], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(coeff_mean[i], color='red', linestyle='--', linewidth=2, label=f'均值: {coeff_mean[i]:.3f}')
                ax.axvline(coeff_median[i], color='orange', linestyle='--', linewidth=2, label=f'中位数: {coeff_median[i]:.3f}')
                ax.set_title(f'模态 {i+1} 系数分布\n(σ={coeff_std[i]:.3f})')
                ax.set_xlabel('系数值')
                ax.set_ylabel('频次')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(k, len(axes)):
                axes[i].axis('off')
            
            # 字体设置
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass
            
            plt.tight_layout()
            coeff_dist_png = self.output_dir / 'coeff_distribution_histogram.png'
            plt.savefig(coeff_dist_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                coeff_dist_svg = self.output_dir / 'coeff_distribution_histogram.svg'
                plt.savefig(coeff_dist_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                coeff_dist_svg = coeff_dist_png
            plt.close(fig)
            output['coeff_dist_png'] = str(coeff_dist_png)
            output['coeff_dist_svg'] = str(coeff_dist_svg)
            logger.info(f"系数分布直方图已保存到: {coeff_dist_png}")
        except Exception as e:
            logger.warning(f"绘制系数分布直方图失败: {e}")
        
        # 2) 重构样本的空间分布统计热力图
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 2.1) 重构样本均值热力图
            vmax_mean = float(np.max(np.abs(recon_mean))) + 1e-12
            vmin_mean = -vmax_mean
            im1 = axes[0].imshow(recon_mean, cmap='RdBu_r', vmin=vmin_mean, vmax=vmax_mean)
            axes[0].set_title('重构样本空间均值分布')
            axes[0].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            cbar1.ax.set_ylabel('Amplitude', rotation=90)
            
            # 2.2) 重构样本标准差热力图  
            im2 = axes[1].imshow(recon_std, cmap='viridis', vmin=0)
            axes[1].set_title('重构样本空间标准差分布')
            axes[1].axis('off')
            cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            cbar2.ax.set_ylabel('标准差', rotation=90)
            
            # 字体设置
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
                    cbar1.ax.yaxis.get_label().set_fontfamily(fam)
                    cbar2.ax.yaxis.get_label().set_fontfamily(fam)
            except Exception:
                pass
                
            plt.tight_layout()
            spatial_dist_png = self.output_dir / 'spatial_distribution_heatmap.png'
            plt.savefig(spatial_dist_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                spatial_dist_svg = self.output_dir / 'spatial_distribution_heatmap.svg'
                plt.savefig(spatial_dist_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                spatial_dist_svg = spatial_dist_png
            plt.close(fig)
            output['spatial_dist_png'] = str(spatial_dist_png)
            output['spatial_dist_svg'] = str(spatial_dist_svg)
            logger.info(f"空间分布热力图已保存到: {spatial_dist_png}")
        except Exception as e:
            logger.warning(f"绘制空间分布热力图失败: {e}")
        
        # 3) 系数矩阵热力图：展示所有样本在各模态上的系数分布模式
        try:
            # 为了更好的可视化效果，如果样本数太多，选择代表性子集
            display_samples = min(n_samples, 100)  # 最多显示100个样本
            if n_samples > display_samples:
                # 选择具有代表性的样本（均匀抽样）
                sample_indices = np.linspace(0, n_samples-1, display_samples, dtype=int)
                display_coeffs = coefficients[sample_indices]
            else:
                display_coeffs = coefficients
                sample_indices = np.arange(n_samples)
                
            fig, ax = plt.subplots(figsize=(max(8, k*0.8), max(6, display_samples*0.05)))
            
            # 生成热力图
            im = ax.imshow(display_coeffs, cmap='RdBu_r', aspect='auto')
            
            # 设置标签和标题
            ax.set_xlabel('模态索引')
            ax.set_ylabel('样本索引')
            ax.set_title(f'系数矩阵分布热力图\n({display_samples}个样本 × {k}个模态)')
            
            # 设置刻度
            ax.set_xticks(range(k))
            ax.set_xticklabels([f'模态{i+1}' for i in range(k)])
            
            if display_samples <= 50:  # 只有在样本数较少时显示Y轴标签
                ax.set_yticks(range(0, display_samples, max(1, display_samples//20)))
                ax.set_yticklabels([f'样本{sample_indices[i]+1}' for i in range(0, display_samples, max(1, display_samples//20))])
            else:
                ax.set_yticks(range(0, display_samples, max(1, display_samples//10)))
                ax.set_yticklabels([f'{sample_indices[i]+1}' for i in range(0, display_samples, max(1, display_samples//10))])
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('系数值', rotation=90)
            
            # 字体设置
            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
                    cbar.ax.yaxis.get_label().set_fontfamily(fam)
            except Exception:
                pass
            
            plt.tight_layout()
            coeff_matrix_png = self.output_dir / 'coeff_matrix_heatmap.png'
            plt.savefig(coeff_matrix_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                coeff_matrix_svg = self.output_dir / 'coeff_matrix_heatmap.svg'
                plt.savefig(coeff_matrix_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                coeff_matrix_svg = coeff_matrix_png
            plt.close(fig)
            output['coeff_matrix_png'] = str(coeff_matrix_png)
            output['coeff_matrix_svg'] = str(coeff_matrix_svg)
            logger.info(f"系数矩阵热力图已保存到: {coeff_matrix_png}")
        except Exception as e:
            logger.warning(f"绘制系数矩阵热力图失败: {e}")

        # 3.5) 模态矩阵值分布直方图（每个模态的矩阵元素值分布）
        try:
            cols = min(4, k)
            rows = (k + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2.5))
            if k == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if isinstance(axes, plt.Axes) else axes
            else:
                axes = axes.ravel()

            # 计算每个模态矩阵的统计量
            mode_means = []
            for i in range(k):
                ax = axes[i] if i < len(axes) else None
                if ax is None:
                    continue
                flat_vals = modes[i].ravel().astype(float)
                ax.hist(flat_vals, bins=30, alpha=0.7, color='seagreen', edgecolor='black')
                ax.axvline(np.mean(flat_vals), color='red', linestyle='--', linewidth=1.5, label=f'均值: {np.mean(flat_vals):.3f}')
                ax.axvline(np.median(flat_vals), color='orange', linestyle='--', linewidth=1.5, label=f'中位数: {np.median(flat_vals):.3f}')
                ax.set_title(f'模态 {i+1} 矩阵值分布')
                ax.set_xlabel('值')
                ax.set_ylabel('频次')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                mode_means.append(np.mean(flat_vals))

            for i in range(k, len(axes)):
                axes[i].axis('off')

            try:
                import matplotlib
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass

            plt.tight_layout()
            mode_matrix_hist_png = self.output_dir / 'mode_matrix_value_histogram.png'
            plt.savefig(mode_matrix_hist_png, dpi=300, bbox_inches='tight')
            try:
                import matplotlib as mpl
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                mode_matrix_hist_svg = self.output_dir / 'mode_matrix_value_histogram.svg'
                plt.savefig(mode_matrix_hist_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                mode_matrix_hist_svg = mode_matrix_hist_png
            plt.close(fig)

            output['mode_matrix_hist_png'] = str(mode_matrix_hist_png)
            output['mode_matrix_hist_svg'] = str(mode_matrix_hist_svg)
            logger.info(f"模态矩阵值分布直方图已保存到: {mode_matrix_hist_png}")
        except Exception as e:
            logger.warning(f"绘制模态矩阵值分布直方图失败: {e}")
        
        # 4) 系数-空间分布的3D可视化 (选择前几个模态)
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            import matplotlib as mpl
            from matplotlib import cm, colors

            # 选择前3个模态进行3D展示
            n_modes_3d = min(3, k)
            fig = plt.figure(figsize=(5*n_modes_3d, 5))
            
            for mode_idx in range(n_modes_3d):
                ax3d = fig.add_subplot(1, n_modes_3d, mode_idx+1, projection='3d')
                
                # 使用该模态的系数分布作为颜色映射
                mode_data = modes[mode_idx]  # [H, W]
                Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                xpos = X.ravel()
                ypos = Y.ravel()
                zpos = np.zeros_like(xpos, dtype=float)
                dx = 0.8 * np.ones_like(xpos, dtype=float)
                dy = 0.8 * np.ones_like(ypos, dtype=float)
                dz = mode_data.ravel().astype(float)

                # 根据系数分布的标准差来调整颜色强度
                coeff_intensity = coeff_std[mode_idx] / (np.max(coeff_std) + 1e-12)
                dmin = float(np.min(dz))
                dmax = float(np.max(dz))
                colors_map = cm.get_cmap('RdBu_r')
                # 根据数据分布选择合适归一化，避免 TwoSlopeNorm 对 vmin/vcenter/vmax 的严格升序要求报错
                if (dmin < 0.0) and (dmax > 0.0):
                    # 数据同时包含正负，使用 TwoSlopeNorm，并确保严格 vmin < 0 < vmax
                    vmin = dmin if dmin < 0.0 else -1e-12
                    vmax = dmax if dmax > 0.0 else 1e-12
                    # 防止接近0导致与vcenter相等
                    if vmin == 0.0:
                        vmin = -1e-12
                    if vmax == 0.0:
                        vmax = 1e-12
                    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                else:
                    # 数据为单侧（全非负或全非正）或常量数组，退化为线性归一化并确保范围非零
                    if dmin == dmax:
                        if dmin == 0.0:
                            vmin, vmax = -1e-6, 1e-6
                        else:
                            delta = max(1e-6, abs(dmin) * 1e-6)
                            vmin, vmax = dmin - delta, dmax + delta
                    else:
                        vmin, vmax = dmin, dmax
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                facecolors = colors_map(norm(dz))
                # 调整透明度反映系数变异性
                facecolors[:, 3] = 0.3 + 0.7 * coeff_intensity  

                ax3d.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=facecolors, edgecolor='k', linewidth=0.1)
                ax3d.set_title(f'模态 {mode_idx+1}\n(系数σ={coeff_std[mode_idx]:.3f})')
                ax3d.set_xlabel('X')
                ax3d.set_ylabel('Y')
                ax3d.set_zlabel('模态值')
                ax3d.view_init(elev=30, azim=-60)

            # 字体设置
            try:
                fam = CHINESE_FONT_NAME
                if fam:
                    for txt in fig.findobj(match=mpl.text.Text):
                        txt.set_fontfamily(fam)
            except Exception:
                pass

            plt.tight_layout()
            mode_dist_3d_png = self.output_dir / 'mode_distribution_3d.png'
            plt.savefig(mode_dist_3d_png, dpi=300, bbox_inches='tight')
            try:
                old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
                mpl.rcParams['svg.fonttype'] = 'none'
                mode_dist_3d_svg = self.output_dir / 'mode_distribution_3d.svg'
                plt.savefig(mode_dist_3d_svg, format='svg', bbox_inches='tight')
                if old_svg_fonttype is not None:
                    mpl.rcParams['svg.fonttype'] = old_svg_fonttype
            except Exception:
                mode_dist_3d_svg = mode_dist_3d_png
            plt.close(fig)
            output['mode_dist_3d_png'] = str(mode_dist_3d_png)
            output['mode_dist_3d_svg'] = str(mode_dist_3d_svg)
            logger.info(f"模态分布3D图已保存到: {mode_dist_3d_png}")
        except Exception as e:
            logger.warning(f"绘制模态分布3D图失败: {e}")
        
        return output


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='分析HDF5文件的输出模态')
    parser.add_argument('data_path', type=str, nargs='?', default=r"X:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5", help='HDF5数据文件路径（留空将自动选择工作区内最大的HDF5文件）')
    parser.add_argument('--output-dir', type=str, default='output_modality_analysis', help='输出目录')
    parser.add_argument('--max-samples', type=int, default=1000, help='最大样本数')
    parser.add_argument('--n-modes', type=int, default=64, help='保留的模态数')
    parser.add_argument('--max-modes-to-save', type=int, default=9, help='最大保存的可视化模态数')
    parser.add_argument('--energy-threshold', type=float, default=0.99, help='能量阈值')
    parser.add_argument('--compute-similarities', action='store_true', help='计算样本相似度')
    parser.add_argument('--similarity-max-show', type=int, default=200, help='相似度热力图最大显示样本数')
    parser.add_argument('--analyze-2d-modes', action='store_true', help='保留2D结构进行模态系数/矩阵统计与叠加态可视化')
    
    # 新增：每模态相似度分析参数
    parser.add_argument('--analyze-per-mode-similarity', action='store_true', help='为每个模态生成样本相似度热力图（每模态一张）')
    parser.add_argument('--per-mode-max-show', type=int, default=200, help='每张每模态热力图最多显示的样本数')
    
    # 新增：样本重构相似度分析参数
    parser.add_argument('--analyze-sample-similarity', action='store_true', help='对每个样本在不同模态数k下的重构结果进行相似度分析')
    parser.add_argument('--similarity-k-range', type=int, nargs='*', default=None, help='相似度分析的k值列表（空则默认[1,2,4,8,16,32,64]内不超过有效模态数的部分）')
    parser.add_argument('--similarity-max-samples', type=int, default=50, help='参与相似度分析的最大样本数')
    
    # 新增：逐样本SVD参数
    parser.add_argument('--per-sample-svd', action='store_true', help='使用逐样本SVD方法：对每个样本的H×W矩阵单独进行SVD，然后对右奇异向量进行平均')
    
    # 新增：系数可视化参数
    parser.add_argument('--visualize-coefficients', action='store_true', help='生成系数的可视化图表：均值柱状图、分布箱线图和相关性热力图')
    parser.add_argument('--coefficient-modes', type=int, default=None, help='可视化的系数模态数，默认为所有模态')
    
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 若未提供数据路径，则自动检索工作区内体积最大的 HDF5 文件
    if args.data_path is None:
        try:
            # 脚本位于 generate_data/utils 下，两级父目录即工作区根目录
            search_root = Path(__file__).resolve().parents[2]
        except Exception:
            search_root = Path.cwd()
        patterns = ['*.h5', '*.hdf5']
        candidates = []
        for pat in patterns:
            try:
                candidates.extend(search_root.rglob(pat))
            except Exception:
                pass
        if not candidates:
            logger.error('未提供数据路径且未在工作区找到任何HDF5文件')
            return
        data_path = max(candidates, key=lambda p: p.stat().st_size)
        logger.info(f"未提供数据路径，自动选择工作区最大HDF5: {data_path} ({data_path.stat().st_size} bytes)")
    else:
        data_path = Path(args.data_path)
    
    try:
        # 创建分析器
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer = OutputModalityAnalyzer(output_dir)
        
        # 1. 加载数据
        logger.info(f"正在加载数据: {data_path}")
        data_dict = analyzer.load_hdf5_dataset(data_path)
        if data_dict is None:
            logger.error("数据加载失败")
            return
        
        # 2. 预处理数据
        logger.info("正在预处理数据...")
        preprocessed_data, spatial_shape = analyzer.preprocess_output_data(
            data_dict['data'], 
            max_samples=args.max_samples
        )
        
        # 3. 执行SVD分析（支持逐样本SVD）
        logger.info("正在执行SVD分析...")
        if getattr(args, 'per_sample_svd', False):
            logger.info("使用逐样本SVD：对每个样本的H×W矩阵单独进行SVD，并对右奇异向量进行平均")
            analysis_results = analyzer.perform_per_sample_svd_analysis(
                preprocessed_data,
                spatial_shape=spatial_shape,
                n_modes=args.n_modes,
                energy_threshold=args.energy_threshold
            )
        else:
            analysis_results = analyzer.perform_svd_analysis(
                preprocessed_data,
                n_modes=args.n_modes,
                energy_threshold=args.energy_threshold
            )
        
        # 4. 可视化分析结果
        logger.info("正在生成可视化...")
        visualization_results = analyzer.visualize_modal_bases(
            analysis_results['Vt_reduced'], 
            spatial_shape,
            max_modes_to_save=args.max_modes_to_save
        )
        analysis_results['visualizations'] = visualization_results

        # 4.2 逐样本SVD专属：系数与模态矩阵分析（仅在 per-sample 模式下）
        if analysis_results.get('config', {}).get('method') == 'per_sample_averaged':
            try:
                coeff_outputs = analyzer.analyze_per_sample_svd_coefficients(analysis_results)
                if coeff_outputs:
                    analysis_results['per_sample_svd_coefficients'] = coeff_outputs
            except Exception as e:
                logger.warning(f"逐样本SVD系数分析失败: {e}")

            # 4.3 逐样本模态可视化
            try:
                per_sample_data = analysis_results.get('per_sample_svd_data')
                if isinstance(per_sample_data, dict):
                    # 新增：按模态分开保存，默认前4个模态，行列自适应排布
                    try:
                        separate_viz = analyzer.visualize_per_sample_modes_separate(
                            per_sample_data,
                            sample_count=50,
                            modes=list(range(min(4, per_sample_data['all_singular_values'].shape[1]))),
                            scale_by_singular_value=True,
                            random=False,
                            max_cols=10
                        )
                        if separate_viz:
                            analysis_results['per_sample_modes_separate'] = separate_viz
                    except Exception as e_sep:
                        logger.warning(f"逐样本分模态可视化失败: {e_sep}")

                    # 保留：原先的两模态横向对比（兼容需求）
                    leading_viz = analyzer.visualize_per_sample_leading_modes(
                        per_sample_data,
                        sample_count=50,
                        modes=(0, 1),
                        scale_by_singular_value=True,
                        random=False
                    )
                    if leading_viz:
                        analysis_results['per_sample_leading_modes'] = leading_viz
                else:
                    logger.warning('未找到 per_sample_svd_data，跳过逐样本模态可视化')
            except Exception as e:
                logger.warning(f"逐样本模态可视化失败: {e}")

        # 4.1 可选：每模态样本相似度热力图
        if args.analyze_per_mode_similarity:
            # 优先使用 N×k 的模态系数矩阵 raw_scores；若不存在则尝试计算；最后回退到 sample_embeddings
            coeff_embeddings = analysis_results.get('raw_scores')
            if coeff_embeddings is None:
                Vt_reduced = analysis_results.get('Vt_reduced')
                if Vt_reduced is not None:
                    try:
                        coeff_embeddings = preprocessed_data @ Vt_reduced.T
                        analysis_results['raw_scores'] = coeff_embeddings
                        logger.info("已计算 raw_scores 作为每模态相似度分析的系数矩阵")
                    except Exception as e:
                        logger.warning(f"计算 raw_scores 失败，回退到 sample_embeddings: {e}")
                else:
                    logger.warning("未找到 Vt_reduced，无法计算 raw_scores，回退到 sample_embeddings")
            if coeff_embeddings is None:
                coeff_embeddings = analysis_results.get('sample_embeddings')
                if coeff_embeddings is None:
                    logger.warning('未找到样本嵌入，跳过每模态相似度分析')
                else:
                    logger.warning('使用 sample_embeddings 进行每模态相似度分析，注意逐样本SVD路线下其维度语义与按模态系数不完全一致')
            if coeff_embeddings is not None:
                n_modes_for_similarity = coeff_embeddings.shape[1]
                per_mode_outputs = analyzer.analyze_per_mode_sample_similarity(
                    coeff_embeddings,
                    n_modes=n_modes_for_similarity,
                    max_show=args.per_mode_max_show
                )
                analysis_results['per_mode_similarity'] = {
                    'heatmaps': {k: str(v) for k, v in per_mode_outputs.items() if 'png' in k or 'svg' in k},
                    'summary': per_mode_outputs.get('summary', {})
                }
        
        # 5. 添加基本信息
        analysis_results.update({
            'output_shape': data_dict['data'].shape,
            'n_samples': preprocessed_data.shape[0],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_shape': preprocessed_data.shape,
            'reduced_shape': analysis_results['sample_embeddings'].shape
        })
        
        # 6. 可选：计算样本相似度
        if args.compute_similarities:
            embeddings = analysis_results.get('sample_embeddings')
            if embeddings is None:
                logger.warning('未找到样本嵌入，跳过相似度计算')
            else:
                similarities = analyzer.compute_sample_similarities(embeddings)
                heatmaps = analyzer.visualize_sample_similarities(similarities, sample_count=analysis_results['n_samples'], max_show=args.similarity_max_show)
                saved_paths = analyzer.save_similarity_matrices(similarities)
                sim_summary = analyzer.summarize_similarities(similarities['cosine_similarity'])
                analysis_results['similarity'] = {
                    'matrices': saved_paths,
                    'heatmaps': {k: str(v) for k, v in heatmaps.items()},
                    # 兼容旧字段（默认返回余弦相似度热力图路径）
                    'heatmap_png': str(heatmaps.get('cosine_png', '')),
                    'heatmap_svg': str(heatmaps.get('cosine_svg', '')),
                    'summary': sim_summary,
                }
        
        # 7. 可选：进行2D模态统计
        if args.analyze_2d_modes:
            logger.info("正在进行2D模态系数与矩阵统计分析...")
            # 使用未去均值的原始数据在模态基上做投影，得到原始系数分布
            Vt_reduced = analysis_results.get('Vt_reduced')
            if Vt_reduced is None:
                logger.warning("未找到Vt_reduced，跳过2D模态统计分析")
            else:
                logger.info("采用未去均值的原始投影系数 raw_scores 进行2D模态分布分析")
                raw_scores = preprocessed_data @ Vt_reduced.T  # [n_samples, k]
                stats_paths = analyzer.analyze_2d_modes(
                    Vt_reduced,
                    raw_scores,
                    spatial_shape
                )
                analysis_results['mode_2d_stats'] = stats_paths
                analysis_results['raw_scores'] = raw_scores  # 记录原始投影系数，便于溯源或后续分析
        
        # 7.1 新增：可选 系数可视化
        if getattr(args, 'visualize_coefficients', False):
            logger.info("正在生成系数可视化图表...")
            # 优先使用原始投影系数 raw_scores，若不存在则使用嵌入 sample_embeddings
            coeff_matrix = analysis_results.get('raw_scores')
            if coeff_matrix is None:
                coeff_matrix = analysis_results.get('sample_embeddings')
            if coeff_matrix is None:
                Vt_reduced = analysis_results.get('Vt_reduced')
                if Vt_reduced is not None:
                    coeff_matrix = preprocessed_data @ Vt_reduced.T
                    analysis_results['raw_scores'] = coeff_matrix
            if coeff_matrix is None:
                logger.warning('未找到可用于系数可视化的矩阵，已跳过系数可视化')
            else:
                coeff_vis = analyzer.visualize_coefficients(coeff_matrix, n_modes=getattr(args, 'coefficient_modes', None))
                analysis_results['coefficients_visualizations'] = coeff_vis
        
        # 7.2 可选：进行样本重构相似度分析（每个样本、每个k）
        if getattr(args, 'analyze_sample_similarity', False):
            logger.info("正在进行样本重构相似度分析...")
            Vt_reduced = analysis_results.get('Vt_reduced')
            raw_scores = analysis_results.get('raw_scores')
            if Vt_reduced is None:
                logger.warning("未找到Vt_reduced，跳过样本相似度分析")
            else:
                if raw_scores is None:
                    raw_scores = preprocessed_data @ Vt_reduced.T
                sim_paths = analyzer.analyze_sample_reconstruction_similarity(
                    original_data=preprocessed_data,
                    Vt_reduced=Vt_reduced,
                    coefficients=raw_scores,
                    spatial_shape=spatial_shape,
                    k_range=args.similarity_k_range,
                    max_samples_to_analyze=args.similarity_max_samples
                )
                analysis_results['reconstruction_similarity'] = sim_paths
        
        # 8. 生成总结报告
        logger.info("正在生成分析报告...")
        analyzer.save_analysis_report(analysis_results, output_dir)
        
        logger.info(f"分析完成！结果保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()