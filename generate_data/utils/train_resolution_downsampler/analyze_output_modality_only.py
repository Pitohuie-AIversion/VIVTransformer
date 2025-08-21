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
        
        # 数据标准化
        output_mean = np.mean(output_data, axis=0, keepdims=True)
        output_std = np.std(output_data, axis=0, keepdims=True) + 1e-8
        output_data_std = (output_data - output_mean) / output_std
        
        logger.info("计算输出数据SVD...")
        U, S, Vt = np.linalg.svd(output_data_std, full_matrices=False)
        
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
        mse = np.mean((output_data_std - reconstructed)**2)
        relative_error = mse / np.mean(output_data_std**2)
        
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
        all_mode_images = []  # 收集每个样本的每个模态的H×W模式图
        
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
            
            # 计算样本嵌入（U @ S的前k个模态）
            sample_embedding = (U_sample_reduced @ np.diag(S_sample_reduced))[:, :effective_modes]
            sample_embeddings_list.append(sample_embedding.flatten())  # 展平为1D
            
            # 计算每个模态的模式图(按S加权的外积)，形状 [effective_modes, H, W]
            mode_images = np.array([
                S_sample_reduced[m] * np.outer(U_sample_reduced[:, m], Vt_sample_reduced[m, :])
                for m in range(effective_modes)
            ])
            all_mode_images.append(mode_images)
        
        # 将所有样本的Vt矩阵和模式图进行平均
        all_Vt = np.array(all_Vt)  # [N, effective_modes, W]
        all_singular_values = np.array(all_singular_values)  # [N, effective_modes]
        all_mode_images = np.array(all_mode_images)  # [N, effective_modes, H, W]
        
        # 对右奇异向量进行平均（保留以便统计需要）
        averaged_Vt = np.mean(all_Vt, axis=0)  # [effective_modes, W]
        averaged_singular_values = np.mean(all_singular_values, axis=0)  # [effective_modes]
        
        # 平均后的模式图，并展平为 (effective_modes, H*W) 以便可视化
        averaged_mode_images = np.mean(all_mode_images, axis=0)  # [effective_modes, H, W]
        Vt_reduced_like = averaged_mode_images.reshape(effective_modes, H * W)
        
        # 计算平均后的能量比和累积能量
        energy_ratios = (averaged_singular_values**2) / np.sum(averaged_singular_values**2) if np.sum(averaged_singular_values**2) > 0 else np.zeros_like(averaged_singular_values)
        cumulative_energy = np.cumsum(energy_ratios)
        
        # 构建样本嵌入矩阵
        sample_embeddings_array = np.array(sample_embeddings_list)  # [N, H*effective_modes]
        
        # 使用平均后的Vt重建数据以计算误差
        reconstructed_samples = []
        for i in range(n_samples):
            # 使用原始样本的U和S，但是平均后的Vt进行重建
            sample_matrix = reshaped_data[i]
            U_sample, S_sample, _ = np.linalg.svd(sample_matrix, full_matrices=False)
            
            # 只使用前effective_modes个模态
            U_reduced = U_sample[:, :effective_modes]
            S_reduced = S_sample[:effective_modes]
            
            # 重建使用平均后的Vt
            reconstructed = U_reduced @ np.diag(S_reduced) @ averaged_Vt
            reconstructed_samples.append(reconstructed.flatten())
        
        # 计算重建误差
        original_flat = output_data  # [N, H*W]
        reconstructed_flat = np.array(reconstructed_samples)  # [N, H*W]
        
        mse = np.mean((original_flat - reconstructed_flat)**2)
        relative_error = mse / (np.mean(original_flat**2) + 1e-12)
        
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
            'Vt_reduced': Vt_reduced_like,  # 用于可视化的平均模式图(展平)
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
                'avg_energy_per_mode': energy_ratios.tolist()
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
        
        logger.info(f"分析报告已保存到: {summary_path}")
        return summary_path


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='分析HDF5文件的输出模态')
    parser.add_argument(
        'data_path',
        type=str,
        nargs='?',
        default=r"X:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\PDEBench\\pdebench\\data_download\\2D_DarcyFlow_beta0.1_Train.hdf5",
        help='HDF5数据文件路径（默认：DarcyFlow 训练集）'
    )
    parser.add_argument('--output-dir', type=str, default='output_modality_analysis', help='输出目录')
    parser.add_argument('--max-samples', type=int, default=1000, help='最大样本数')
    parser.add_argument('--n-modes', type=int, default=64, help='保留的模态数')
    parser.add_argument('--max-modes-to-save', type=int, default=9, help='最大保存的可视化模态数')
    parser.add_argument('--energy-threshold', type=float, default=0.99, help='能量阈值')
    parser.add_argument('--compute-similarities', action='store_true', help='计算样本相似度')
    parser.add_argument('--similarity-max-show', type=int, default=200, help='相似度热力图最大显示样本数')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--per-sample-svd', action='store_true', help='逐样本SVD（对每个样本的H×W进行SVD并平均空间基）')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # 创建分析器
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        analyzer = OutputModalityAnalyzer(output_dir)
        
        # 1. 加载数据
        logger.info(f"正在加载数据: {args.data_path}")
        data_dict = analyzer.load_hdf5_dataset(Path(args.data_path))
        if data_dict is None:
            logger.error("数据加载失败")
            return
        
        # 2. 预处理数据
        logger.info("正在预处理数据...")
        preprocessed_data, spatial_shape = analyzer.preprocess_output_data(
            data_dict['data'], 
            max_samples=args.max_samples
        )
        
        # 3. 执行SVD分析
        if args.per_sample_svd:
            logger.info("正在执行逐样本SVD分析...")
            analysis_results = analyzer.perform_per_sample_svd_analysis(
                preprocessed_data,
                spatial_shape,
                n_modes=args.n_modes,
                energy_threshold=args.energy_threshold
            )
        else:
            logger.info("正在执行SVD分析...")
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
        # 新增：系数相关可视化（均值柱状图、箱线图、热力图）
        try:
            coeff_vis = analyzer.visualize_coefficients(
                analysis_results,
                max_modes_to_show=args.n_modes,
                max_samples_to_show=max(200, args.similarity_max_show)
            )
            visualization_results.update(coeff_vis)
        except Exception as e:
            logger.warning(f"系数可视化生成失败: {e}")
        analysis_results['visualizations'] = visualization_results
        
        # 5. 添加基本信息
        analysis_results.update({
            'output_shape': data_dict['data'].shape,
            'n_samples': preprocessed_data.shape[0],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_shape': preprocessed_data.shape,
            'reduced_shape': (analysis_results['sample_embeddings'].shape if 'sample_embeddings' in analysis_results else analysis_results['U_reduced'].shape)
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
        
        # 7. 生成总结报告
        logger.info("正在生成分析报告...")
        analyzer.save_analysis_report(analysis_results, output_dir)
        
        logger.info(f"分析完成！结果保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()