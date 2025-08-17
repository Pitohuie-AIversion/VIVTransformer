#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的输出模态分析脚本
仅对数据集的 tensor 数据进行 SVD 模态分解，不涉及输入-输出配对和潜空间投影

功能:
1. 加载 HDF5 文件中的 tensor 数据集
2. 对展平后的 tensor 数据进行 SVD 分解
3. 分析奇异值分布和能量保持
4. 可视化模态结构（仅输出相关）
5. 生成简化的模态分析报告

Usage:
    python analyze_output_modality_only.py [--data-path path/to/data.h5] [--n-modes 64] [--output-dir results]
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

# 预先设置警告过滤器
import warnings
warnings.filterwarnings("ignore", message=r".*missing from font.*")
warnings.filterwarnings("ignore", message=r".*does not have a glyph.*")
warnings.filterwarnings("ignore", message=r".*substituting with a dummy symbol.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 记录选中的中文字体家族名称，便于绘图阶段强制应用
CHINESE_FONT_NAME = None


def configure_chinese_font():
    """配置中文字体支持（优先项目内字体，确保 PNG 正常显示中文）"""
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as fm
        import os

        # 声明使用全局变量用于在绘图阶段读取
        global CHINESE_FONT_NAME

        # 注册项目内字体（优先使用思源黑体 SC，其次 Noto Sans SC）
        proj_name = None
        for pf in [
            r"x:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\fonts\\SourceHanSansSC-Regular.otf",
            r"x:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\fonts\\NotoSansSC-Regular.otf",
        ]:
            if os.path.exists(pf):
                try:
                    fm.fontManager.addfont(pf)
                    proj_name = fm.FontProperties(fname=pf).get_name()
                    logger.info(f"已注册项目字体: {proj_name} ({pf})")
                    break
                except Exception as e:
                    logger.warning(f"项目字体注册失败: {pf}: {e}")
                    proj_name = None

        # 也尝试注册常见中文字体（Windows）
        for fp in [
            r"C:\\Windows\\Fonts\\msyh.ttc",
            r"C:\\Windows\\Fonts\\msyh.ttf",
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:\\Windows\\Fonts\\simsun.ttc",
        ]:
            if os.path.exists(fp):
                try:
                    fm.fontManager.addfont(fp)
                    logger.info(f"已注册系统字体: {fp}")
                except Exception:
                    pass

        # 刷新字体列表
        try:
            available_names = {f.name for f in fm.fontManager.ttflist}
        except Exception:
            available_names = set()

        # 直接使用在系统中能找到的字体，优先级从高到低
        candidate_fonts = [
            "Microsoft YaHei",  # 微软雅黑，Win7+ 标配
            "SimHei",          # 黑体，Windows 自带
            "Microsoft YaHei UI",
            "SimSun",          # 宋体，Windows 自带
            "Arial Unicode MS",
            "DejaVu Sans",     # 回退选项
        ]
        if proj_name:
            candidate_fonts.insert(0, proj_name)  # 项目字体放首位（如果可用）

        selected = None
        for name in candidate_fonts:
            if name in available_names:
                selected = name
                break
        if selected is None:
            selected = "SimHei"  # 最后兜底
        # 记录最终选中的中文字体家族（写入全局）
        CHINESE_FONT_NAME = proj_name or selected

        # 将具体字体名设为首选 family，避免 SVG 写入通用 'sans-serif' 导致首位变为 Arial
        if proj_name:
            mpl.rcParams["font.family"] = proj_name
            mpl.rcParams["font.sans-serif"] = [proj_name] + [f for f in candidate_fonts if f != proj_name]
        else:
            mpl.rcParams["font.family"] = selected
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

    def preprocess_output_data(self, data: np.ndarray, max_samples: int = 1000) -> np.ndarray:
        """预处理输出数据 - 将多维数据展平为2D矩阵"""
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
            return None
        
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
        
        return output_data

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
        output_reconstructed = (U_reduced * S_reduced) @ Vt_reduced
        
        # 计算误差
        mse = np.mean((output_data_std - output_reconstructed)**2)
        relative_error = np.sqrt(mse) / (np.std(output_data_std) + 1e-8)
        
        logger.info(f"重建误差: MSE={mse:.6f}, 相对误差={relative_error:.6f}")
        
        # 计算降维后的样本嵌入向量（用于后续相似度分析）
        sample_embeddings = U_reduced * S_reduced  # [n_samples, effective_modes]
        logger.info(f"生成样本嵌入向量: {sample_embeddings.shape}")
        
        # 整理结果
        analysis_results = {
            'singular_values': S,
            'energy_ratios': (S**2) / np.sum(S**2),
            'cumulative_energy': cumulative_energy,
            'effective_modes': effective_modes,
            'output_dim': output_dim,
            'n_samples': n_samples,
            'sample_embeddings': sample_embeddings,  # 新增：样本嵌入向量
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

    def visualize_analysis(self, analysis_results: Dict):
        """可视化模态分析结果"""
        logger.info("生成模态分析可视化...")
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('输出数据模态分析结果', fontsize=16, fontweight='bold')
        
        # 1. 奇异值分布
        ax = axes[0, 0]
        n_show = min(50, len(analysis_results['singular_values']))
        ax.semilogy(range(1, n_show+1), analysis_results['singular_values'][:n_show], 
                   'b-o', label='奇异值', markersize=4)
        ax.set_xlabel('模态索引')
        ax.set_ylabel('奇异值 (对数尺度)')
        ax.set_title('奇异值衰减曲线')
        ax.grid(True, alpha=0.3)
        
        # 标记有效模态数
        effective_modes = analysis_results['effective_modes']
        if effective_modes <= n_show:
            ax.axvline(x=effective_modes, color='red', linestyle='--', alpha=0.7, label=f'有效模态数: {effective_modes}')
            ax.legend()
        
        # 2. 能量比例
        ax = axes[0, 1]
        n_show = min(30, len(analysis_results['energy_ratios']))
        indices = np.arange(1, n_show + 1)
        ax.bar(indices, analysis_results['energy_ratios'][:n_show], 
               alpha=0.8, color='blue')
        ax.set_xlabel('模态索引')
        ax.set_ylabel('能量比例')
        ax.set_title('各模态能量贡献')
        ax.set_xticks(indices[::5])  # 每5个显示一个刻度
        ax.grid(True, alpha=0.3)
        
        # 3. 累积能量
        ax = axes[1, 0]
        ax.plot(range(1, len(analysis_results['cumulative_energy'])+1), 
                analysis_results['cumulative_energy'], 'b-', linewidth=2)
        ax.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='90%能量线')
        ax.axhline(y=0.95, color='k', linestyle=':', alpha=0.5, label='95%能量线')
        ax.axhline(y=0.99, color='k', linestyle='-.', alpha=0.5, label='99%能量线')
        
        # 标记有效模态数对应的能量
        energy_at_effective = analysis_results['cumulative_energy'][effective_modes-1]
        ax.axvline(x=effective_modes, color='red', linestyle='--', alpha=0.7)
        ax.plot(effective_modes, energy_at_effective, 'ro', markersize=8, 
                label=f'选择点: {effective_modes}模态, {energy_at_effective:.3f}能量')
        
        ax.set_xlabel('模态数量')
        ax.set_ylabel('累积能量比例')
        ax.set_title('累积能量保持曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # 4. 统计信息
        ax = axes[1, 1]
        ax.axis('off')
        
        # 计算关键统计量
        energy_90 = np.argmax(analysis_results['cumulative_energy'] >= 0.9) + 1
        energy_95 = np.argmax(analysis_results['cumulative_energy'] >= 0.95) + 1
        energy_99 = np.argmax(analysis_results['cumulative_energy'] >= 0.99) + 1
        
        compression_ratio = analysis_results['output_dim'] / effective_modes
        
        stats_text = f"""分析统计:
- 输出维度: {analysis_results['output_dim']:,}
- 样本数量: {analysis_results['n_samples']:,}
- 有效模态数: {effective_modes}
- 压缩比: {compression_ratio:.1f}x

能量保持:
- 90%能量需要: {energy_90} 个模态
- 95%能量需要: {energy_95} 个模态
- 99%能量需要: {energy_99} 个模态

重建误差:
- MSE: {analysis_results['reconstruction_error']['mse']:.6f}
- 相对误差: {analysis_results['reconstruction_error']['relative_error']:.6f}

配置:
- 目标模态数: {analysis_results['config']['n_modes']}
- 能量阈值: {analysis_results['config']['energy_threshold']}"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        # 在保存前，强制将所有文本对象的字体设置为选中的中文字体，避免 SVG 中写入 'Arial'
        try:
            import matplotlib as mpl
            import matplotlib
            fam = CHINESE_FONT_NAME
            if fam:
                # 设置 suptitle 字体
                if getattr(fig, "_suptitle", None) is not None:
                    fig._suptitle.set_fontfamily(fam)
                # 遍历各子图的文本对象
                for ax in axes.ravel():
                    for txt in ax.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
            else:
                logger.warning("未检测到 CHINESE_FONT_NAME，全局强制字体设置可能失效")
        except Exception as e:
            logger.warning(f"强制应用中文字体失败: {e}")
        # 保存图像
        output_path = self.output_dir / 'output_modality_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"模态分析图像已保存到: {output_path}")
        
        # 同时导出两份 SVG：一种保持文本（便于编辑），一种路径化（跨平台不丢字形）
        try:
            import matplotlib as mpl
            old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
            # 文本版（保留字体引用）
            mpl.rcParams['svg.fonttype'] = 'none'
            svg_text_path = self.output_dir / 'output_modality_analysis.svg'
            plt.savefig(svg_text_path, format='svg', bbox_inches='tight')
            logger.info(f"模态分析SVG(文本)已保存到: {svg_text_path}")
            # 路径版（不依赖系统字体）
            mpl.rcParams['svg.fonttype'] = 'path'
            svg_pathified_path = self.output_dir / 'output_modality_analysis_path.svg'
            plt.savefig(svg_pathified_path, format='svg', bbox_inches='tight')
            logger.info(f"模态分析SVG(路径)已保存到: {svg_pathified_path}")
        finally:
            # 还原 svg.fonttype
            if old_svg_fonttype is not None:
                mpl.rcParams['svg.fonttype'] = old_svg_fonttype
        
        plt.close()
        return output_path

    def save_similarity_matrices(self, similarities: Dict) -> Dict[str, str]:
        """将相似度矩阵保存为 .npy 文件，返回保存路径"""
        cos_path = self.output_dir / 'cosine_similarity.npy'
        euc_path = self.output_dir / 'euclidean_distance.npy'
        np.save(cos_path, similarities['cosine_similarity'])
        np.save(euc_path, similarities['euclidean_distance'])
        logger.info(f"相似度矩阵已保存: {cos_path}, {euc_path}")
        return {
            'cosine_similarity': str(cos_path),
            'euclidean_distance': str(euc_path),
        }

    def summarize_similarities(self, cosine_sim: np.ndarray, top_k: int = 5) -> Dict:
        """对余弦相似度矩阵做摘要（排除对角与重复对）"""
        N = cosine_sim.shape[0]
        # 排除对角线，取上三角
        iu = np.triu_indices(N, k=1)
        vals = cosine_sim[iu]
        if vals.size == 0:
            return {'avg': None, 'min': None, 'max': None, 'top_pairs': []}
        avg = float(np.mean(vals))
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        # 选取前 top_k 对
        top_idx = np.argsort(vals)[-top_k:][::-1]
        pairs = []
        for rank, idx in enumerate(top_idx, 1):
            i = int(iu[0][idx]); j = int(iu[1][idx])
            pairs.append({'rank': rank, 'i': i, 'j': j, 'cosine': float(vals[idx])})
        return {'avg': avg, 'min': vmin, 'max': vmax, 'top_pairs': pairs}

    def compute_sample_similarities(self, embeddings: np.ndarray) -> Dict:
        """计算样本间相似度矩阵（余弦相似度与欧氏距离）"""
        logger.info("计算样本相似度矩阵...")
        # 归一化用于余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        emb_norm = embeddings / norms
        cosine_sim = emb_norm @ emb_norm.T  # [N, N]
        # 欧氏距离（在嵌入空间）
        sq = np.sum(embeddings**2, axis=1, keepdims=True)
        euclidean_dist_sq = sq + sq.T - 2 * (embeddings @ embeddings.T)
        euclidean_dist_sq = np.maximum(euclidean_dist_sq, 0.0)
        euclidean_dist = np.sqrt(euclidean_dist_sq)
        logger.info("相似度矩阵计算完成")
        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
        }

    def visualize_sample_similarities(self, similarities: Dict, sample_count: int, max_show: int = 200) -> Tuple[Path, Path]:
        """可视化样本相似度热力图（最多显示 max_show x max_show 以避免过大图像）"""
        logger.info("可视化样本相似度热力图...")
        N = sample_count
        show = min(N, max_show)
        idx = slice(0, show)
        cos_mat = similarities['cosine_similarity'][idx, idx]
        euc_mat = similarities['euclidean_distance'][idx, idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('样本间模态相似度', fontsize=16, fontweight='bold')

        # 1) 余弦相似度热力图（高=相似）
        ax = axes[0]
        sns.heatmap(cos_mat, ax=ax, cmap='viridis', vmin=-1.0, vmax=1.0)
        ax.set_title(f'余弦相似度 (显示前 {show} 个样本)')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('样本索引')

        # 2) 欧氏距离热力图（低=相似）
        ax = axes[1]
        sns.heatmap(euc_mat, ax=ax, cmap='magma')
        ax.set_title(f'欧氏距离 (显示前 {show} 个样本)')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('样本索引')

        plt.tight_layout()
        # 强制中文字体
        try:
            import matplotlib as mpl
            import matplotlib
            fam = CHINESE_FONT_NAME
            if fam and getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_fontfamily(fam)
            if fam:
                for ax in axes.ravel():
                    for txt in ax.findobj(match=matplotlib.text.Text):
                        txt.set_fontfamily(fam)
        except Exception as e:
            logger.warning(f"相似度图应用中文字体失败: {e}")

        png_path = self.output_dir / 'output_sample_similarity.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        logger.info(f"样本相似度PNG已保存到: {png_path}")

        # 导出两份 SVG
        try:
            import matplotlib as mpl
            old_svg_fonttype = mpl.rcParams.get('svg.fonttype', None)
            mpl.rcParams['svg.fonttype'] = 'none'
            svg_text_path = self.output_dir / 'output_sample_similarity.svg'
            plt.savefig(svg_text_path, format='svg', bbox_inches='tight')
            logger.info(f"样本相似度SVG(文本)已保存到: {svg_text_path}")
            mpl.rcParams['svg.fonttype'] = 'path'
            svg_pathified_path = self.output_dir / 'output_sample_similarity_path.svg'
            plt.savefig(svg_pathified_path, format='svg', bbox_inches='tight')
            logger.info(f"样本相似度SVG(路径)已保存到: {svg_pathified_path}")
        finally:
            if 'mpl' in locals() and old_svg_fonttype is not None:
                mpl.rcParams['svg.fonttype'] = old_svg_fonttype

        plt.close()
        return png_path, svg_text_path

    def generate_report(self, analysis_results: Dict, dataset_info: Dict):
        """生成分析报告"""
        logger.info("生成模态分析报告...")
        
        # 计算关键统计量
        effective_modes = analysis_results['effective_modes']
        energy_90 = np.argmax(analysis_results['cumulative_energy'] >= 0.9) + 1
        energy_95 = np.argmax(analysis_results['cumulative_energy'] >= 0.95) + 1
        energy_99 = np.argmax(analysis_results['cumulative_energy'] >= 0.99) + 1
        compression_ratio = analysis_results['output_dim'] / effective_modes
        
        dataset_meta = {
            'file_path': dataset_info.get('file_path'),
            'dataset_info': dataset_info.get('dataset_info', {})
        }
        
        # 为了避免JSON过大，将 sample_embeddings 从报告中简化为形状说明
        mod_analysis = dict(analysis_results)
        if 'sample_embeddings' in mod_analysis:
            try:
                emb_shape = list(mod_analysis['sample_embeddings'].shape)
            except Exception:
                emb_shape = None
            mod_analysis['sample_embeddings'] = {
                'shape': emb_shape,
                'note': '为避免报告过大，未在JSON中展开嵌入矩阵'
            }
        
        report = {
            'dataset_info': dataset_meta,
            'analysis_type': 'output_only',
            'modality_analysis': mod_analysis,
            'summary': {
                'effective_modes': effective_modes,
                'compression_ratio': compression_ratio,
                'energy_90': energy_90,
                'energy_95': energy_95,
                'energy_99': energy_99,
                'reconstruction_error': analysis_results['reconstruction_error']['relative_error']
            }
        }
        
        # 将包含numpy数组/标量的数据转换为可JSON序列化的Python类型
        def _to_json_safe(o):
            import numpy as _np
            if isinstance(o, _np.ndarray):
                return o.tolist()
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, dict):
                return {k: _to_json_safe(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_to_json_safe(x) for x in o]
            return o
        
        # 保存JSON报告
        report_path = self.output_dir / 'output_modality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(_to_json_safe(report), f, indent=2, ensure_ascii=False)
        
        logger.info(f"模态分析报告已保存到: {report_path}")
        
        # 生成markdown摘要
        markdown_path = self.output_dir / 'output_modality_summary.md'
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# 输出数据模态分析报告\n\n")
            f.write("## 数据集信息\n")
            f.write(f"- 文件路径: `{dataset_meta['file_path']}`\n")
            for key, info in dataset_meta['dataset_info'].items():
                f.write(f"- {key}: 形状={info['shape']}, 大小={info['size_mb']:.2f}MB\n")
            
            f.write("\n## 模态分析结果\n")
            f.write(f"- **分析类型**: 仅输出模态分析\n")
            f.write(f"- **有效模态数**: {effective_modes}\n")
            f.write(f"- **压缩比**: {compression_ratio:.1f}x\n")
            f.write(f"- **重建相对误差**: {analysis_results['reconstruction_error']['relative_error']:.6f}\n")
            
            f.write("\n## 能量保持分析\n")
            f.write(f"- **90%能量所需模态数**: {energy_90}\n")
            f.write(f"- **95%能量所需模态数**: {energy_95}\n")
            f.write(f"- **99%能量所需模态数**: {energy_99}\n")
            
            f.write("\n## 主要发现\n")
            if effective_modes < 32:
                f.write("- 数据具有很强的低维结构，可以用较少的模态表示\n")
            elif effective_modes > 128:
                f.write("- 数据复杂度较高，需要较多模态来保持信息\n")
            else:
                f.write("- 数据复杂度适中\n")
                
            if analysis_results['reconstruction_error']['relative_error'] < 0.1:
                f.write("- 重建质量很好，SVD能够很好地捕获数据结构\n")
            else:
                f.write("- 重建误差较大，可能需要更多模态或检查数据质量\n")
            
            # 新增：相似度摘要
            if 'similarity' in analysis_results:
                sim = analysis_results['similarity']
                summ = sim.get('summary', {})
                def _fmt(v):
                    return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"
                f.write("\n## 样本间模态相似度\n")
                f.write(f"- 余弦相似度 平均: {_fmt(summ.get('avg'))}, 最高: {_fmt(summ.get('max'))}, 最低: {_fmt(summ.get('min'))}\n")
                # 列出Top相似样本对
                top_pairs = summ.get('top_pairs', [])
                if top_pairs:
                    f.write("- Top相似样本对:\n")
                    for p in top_pairs:
                        f.write(f"  - Top{p['rank']}: 样本({p['i']}, {p['j']}) 余弦相似度={p['cosine']:.3f}\n")
                # 文件路径
                f.write(f"- 相似度热力图PNG: {sim.get('heatmap_png', 'N/A')}\n")
                f.write(f"- 相似度热力图SVG: {sim.get('heatmap_svg', 'N/A')}\n")
                mats = sim.get('matrices', {})
                if mats:
                    f.write(f"- 矩阵文件: Cosine={mats.get('cosine_similarity', 'N/A')}, Euclidean={mats.get('euclidean_distance', 'N/A')}\n")
        
        logger.info(f"模态分析摘要已保存到: {markdown_path}")
        return report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HDF5数据集输出模态分析（仅分析tensor数据集）')
    parser.add_argument('--data-path', type=str, 
                       default=r'x:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5',
                       help='HDF5数据文件路径')
    parser.add_argument('--n-modes', type=int, default=64, help='SVD模态数量')
    parser.add_argument('--max-samples', type=int, default=1000, help='最大样本数量')
    parser.add_argument('--output-dir', type=str, default='output_modality_analysis', help='输出目录')
    parser.add_argument('--energy-threshold', type=float, default=0.99, help='能量阈值，用于自动选择模态数')
    parser.add_argument('--compute-similarities', action='store_true', help='计算样本之间的模态相似度并生成热力图')
    parser.add_argument('--similarity-max-show', type=int, default=200, help='相似度热图最多显示的样本数')
    
    args = parser.parse_args()
    
    # 设置路径
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    
    logger.info("=== 输出数据模态分析开始 ===")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"分析模态数: {args.n_modes}")
    
    # 创建分析器
    analyzer = OutputModalityAnalyzer(output_dir)
    
    try:
        # 1. 加载数据集
        dataset_info = analyzer.load_hdf5_dataset(data_path)
        if dataset_info is None:
            logger.error("数据集加载失败")
            return
        
        # 2. 预处理数据
        output_data = analyzer.preprocess_output_data(dataset_info['data'], args.max_samples)
        if output_data is None:
            logger.error("数据预处理失败")
            return
        
        logger.info(f"实际进行SVD分析的数据对象:")
        logger.info(f"  - 数据来源: tensor 数据集")
        logger.info(f"  - 数据形状: {output_data.shape}")
        logger.info(f"  - 数据类型: {output_data.dtype}")
        logger.info(f"  - 空间维度: {output_data.shape[1]} (展平后)")
        
        # 3. 进行SVD模态分析
        analysis_results = analyzer.perform_svd_analysis(
            output_data,
            n_modes=args.n_modes,
            energy_threshold=args.energy_threshold
        )
        
        # 3.1 可选：计算样本相似度
        if args.compute_similarities:
            embeddings = analysis_results.get('sample_embeddings')
            if embeddings is None:
                logger.warning('未找到样本嵌入，跳过相似度计算')
            else:
                similarities = analyzer.compute_sample_similarities(embeddings)
                heat_png, heat_svg = analyzer.visualize_sample_similarities(similarities, sample_count=analysis_results['n_samples'], max_show=args.similarity_max_show)
                saved_paths = analyzer.save_similarity_matrices(similarities)
                sim_summary = analyzer.summarize_similarities(similarities['cosine_similarity'])
                analysis_results['similarity'] = {
                    'matrices': saved_paths,
                    'heatmap_png': str(heat_png),
                    'heatmap_svg': str(heat_svg),
                    'summary': sim_summary,
                }
        
        # 4. 生成可视化
        analyzer.visualize_analysis(analysis_results)
        
        # 5. 生成分析报告
        analyzer.generate_report(analysis_results, dataset_info)
        
        logger.info("=== 模态分析完成 ===")
        logger.info(f"结果保存在: {output_dir}")
        logger.info(f"主要发现:")
        logger.info(f"  - 有效模态数: {analysis_results['effective_modes']}")
        logger.info(f"  - 压缩比: {analysis_results['output_dim'] / analysis_results['effective_modes']:.1f}x")
        logger.info(f"  - 重建相对误差: {analysis_results['reconstruction_error']['relative_error']:.6f}")
        if args.compute_similarities and 'similarity' in analysis_results:
            s = analysis_results['similarity']['summary']
            def _fmt(v):
                try:
                    return f"{float(v):.3f}"
                except Exception:
                    return "N/A"
            logger.info(f"  - 相似度摘要: 平均={_fmt(s.get('avg'))}, 最高={_fmt(s.get('max'))}, 最低={_fmt(s.get('min'))}")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()