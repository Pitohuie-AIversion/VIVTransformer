#!/usr/bin/env python3
"""
HDF5数据集模态分析脚本
使用SVD模态投影器分析PDEBench HDF5数据集的模态结构

功能:
1. 加载和分析HDF5数据集
2. 进行SVD模态分解
3. 分析奇异值分布和能量保持
4. 可视化模态结构
5. 生成模态分析报告

Usage:
    python analyze_hdf5_modality.py [--data-path path/to/data.h5] [--n-modes 64] [--output-dir results]
"""

import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns
from pathlib import Path
import sys
import os
import h5py
import json
from typing import Dict, List, Tuple, Optional

# -*- coding: utf-8 -*-
"""
HDF5数据集模态分析工具

使用SVD模态投影器分析PDEBench HDF5数据集的模态结构。
支持：
- 加载HDF5数据集
- SVD模态分解
- 分析奇异值分布和能量保持
- 可视化模态结构
- 生成模态分析报告
"""

# 预先设置警告过滤器（在任何matplotlib导入之前）
import warnings
warnings.filterwarnings("ignore", message=r".*missing from font.*")
warnings.filterwarnings("ignore", message=r".*does not have a glyph.*")
warnings.filterwarnings("ignore", message=r".*substituting with a dummy symbol.*")

import os
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import h5py
import json
import matplotlib

# 添加父目录到 Python 路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入必要模块
from pde_process.svd_modal_projection import SVDModalProjector, SVDProjectionConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HDF5ModalityAnalyzer:
    """HDF5数据集模态分析器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_results = {}
        
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
    
    def preprocess_data(self, data: np.ndarray, max_samples: int = 1000, downsample_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """预处理数据，为SVD分析做准备
        downsample_factor: 当时间维为1或二维数据时，使用分块平均生成低分辨率输入，输入维度为 (downsample_factor^2)
        """
        logger.info(f"预处理数据，原始形状: {data.shape}")
        logger.info(f"使用下采样因子 downsample_factor={downsample_factor}")
        
        # 限制样本数量以控制计算复杂度
        if len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = data[indices]
            logger.info(f"随机选择 {max_samples} 个样本")
        
        # 处理不同的数据形状
        if len(data.shape) == 4:  # [samples, time, height, width]
            n_samples, n_time, height, width = data.shape
            logger.info(f"4D数据: {n_samples}样本, {n_time}时间步, {height}x{width}空间")
            
            # 展平空间维度
            data_flat = data.reshape(n_samples * n_time, height * width)
            
            # 创建输入输出对：使用时间序列的相邻步骤
            if n_time > 1:
                input_indices = np.arange(0, len(data_flat) - 1)
                output_indices = np.arange(1, len(data_flat))
                input_data = data_flat[input_indices]
                output_data = data_flat[output_indices]
            else:
                # 如果只有一个时间步，使用分块平均生成低分辨率输入
                # 注意：这里生成的是 (downsample_factor x downsample_factor) 的网格特征
                ds = max(2, int(downsample_factor))
                input_h, input_w = height // ds, width // ds
                input_data = data.reshape(n_samples, height, width)
                input_data = np.array([
                    np.mean(sample.reshape(height//input_h, input_h, width//input_w, input_w), axis=(1,3))
                    for sample in input_data
                ])
                input_data = input_data.reshape(n_samples, -1)
                output_data = data_flat
                logger.info(f"下采样后输入特征为 {ds}x{ds} 网格，共 {input_data.shape[1]} 维")
                
        elif len(data.shape) == 3:  # [samples, height, width]
            n_samples, height, width = data.shape
            logger.info(f"3D数据: {n_samples}样本, {height}x{width}空间")
            
            # 创建多分辨率输入输出对（分块平均），输入维度为 ds^2
            ds = max(2, int(downsample_factor))
            input_h, input_w = height // ds, width // ds
            input_data = np.array([
                np.mean(sample.reshape(height//input_h, input_h, width//input_w, input_w), axis=(1,3))
                for sample in data
            ])
            input_data = input_data.reshape(n_samples, -1)
            output_data = data.reshape(n_samples, -1)
            logger.info(f"下采样后输入特征为 {ds}x{ds} 网格，共 {input_data.shape[1]} 维")
            
        else:
            logger.error(f"不支持的数据形状: {data.shape}")
            return None, None
        
        logger.info(f"预处理完成:")
        logger.info(f"  输入数据形状: {input_data.shape}")
        logger.info(f"  输出数据形状: {output_data.shape}")
        
        return input_data, output_data
    
    def perform_svd_analysis(self, input_data: np.ndarray, output_data: np.ndarray, 
                           n_modes: int = 64,
                           energy_threshold: float = 0.99,
                           auto_select_modes: bool = True,
                           min_modes: int = 8,
                           max_modes: Optional[int] = None) -> Dict:
        """进行SVD模态分析"""
        logger.info("开始SVD模态分析...")
        
        # 计算最大可用模态数
        inferred_max_modes = min(512, min(input_data.shape[1], output_data.shape[1]))
        if max_modes is None or max_modes <= 0:
            max_modes_final = inferred_max_modes
        else:
            max_modes_final = min(max_modes, inferred_max_modes)
        logger.info(f"允许的最大模态数上限: {max_modes_final} (可用上限: {inferred_max_modes})")
        
        # 创建SVD配置
        config = SVDProjectionConfig(
            n_modes=n_modes,
            energy_threshold=energy_threshold,
            auto_select_modes=auto_select_modes,
            standardize_data=True,
            min_modes=min_modes,
            max_modes=max_modes_final
        )
        
        # 创建并拟合SVD投影器
        projector = SVDModalProjector(config)
        projector.fit(input_data, output_data)
        
        # 获取分析结果
        projection_info = projector.get_projection_info()
        reconstruction_error = projector.compute_reconstruction_error(input_data, output_data)
        
        # 计算能量比例
        input_energy_ratios = (projector.input_singular_values ** 2) / np.sum(projector.input_singular_values ** 2)
        output_energy_ratios = (projector.output_singular_values ** 2) / np.sum(projector.output_singular_values ** 2)
        
        # 计算累积能量
        input_cumulative_energy = np.cumsum(input_energy_ratios)
        output_cumulative_energy = np.cumsum(output_energy_ratios)
        
        analysis_results = {
            'projection_info': projection_info,
            'reconstruction_error': reconstruction_error,
            'input_singular_values': projector.input_singular_values.tolist(),
            'output_singular_values': projector.output_singular_values.tolist(),
            'input_energy_ratios': input_energy_ratios.tolist(),
            'output_energy_ratios': output_energy_ratios.tolist(),
            'input_cumulative_energy': input_cumulative_energy.tolist(),
            'output_cumulative_energy': output_cumulative_energy.tolist(),
            'config': config.to_dict()
        }
        
        # 保存投影器
        projector_path = self.output_dir / 'svd_projector.pkl'
        projector.save(str(projector_path))
        logger.info(f"SVD投影器已保存到: {projector_path}")
        
        return analysis_results, projector

    # 新增：参数扫描（n_modes与能量阈值）
    def scan_svd_parameters(self,
                            input_data: np.ndarray,
                            output_data: np.ndarray,
                            n_modes_list: Optional[List[int]] = None,
                            energy_threshold_list: Optional[List[float]] = None,
                            min_modes: int = 8,
                            max_modes: Optional[int] = None,
                            downsample_factor: Optional[int] = None) -> Dict:
        """批量扫描SVD参数组合并汇总结果"""
        from datetime import datetime
        scan_summary = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'n_modes_scan': [],
            'energy_scan': []
        }
        
        # 扫描 n_modes（禁用自动选择）
        if n_modes_list:
            logger.info(f"开始扫描不同的 n_modes: {n_modes_list}")
            for n in n_modes_list:
                try:
                    results, _ = self.perform_svd_analysis(
                        input_data, output_data,
                        n_modes=int(n),
                        energy_threshold=0.99,
                        auto_select_modes=False,
                        min_modes=min_modes,
                        max_modes=max_modes,
                    )
                    info = results['projection_info']
                    errs = results['reconstruction_error']
                    scan_summary['n_modes_scan'].append({
                        'n_modes': int(n),
                        'latent_dim': info['latent_dim'],
                        'input_dim': info['input_dim'],
                        'output_dim': info['output_dim'],
                        'input_compression_ratio': info['input_dim'] / max(1, info['latent_dim']),
                        'output_compression_ratio': info['output_dim'] / max(1, info['latent_dim']),
                        'input_relative_error': float(errs['input_relative_error']),
                        'output_relative_error': float(errs['output_relative_error'])
                    })
                except Exception as e:
                    logger.error(f"n_modes={n} 扫描失败: {e}")
        
        # 扫描 energy_threshold（启用自动选择）
        if energy_threshold_list:
            logger.info(f"开始扫描不同的能量阈值: {energy_threshold_list}")
            for thr in energy_threshold_list:
                try:
                    results, _ = self.perform_svd_analysis(
                        input_data, output_data,
                        n_modes=64,
                        energy_threshold=float(thr),
                        auto_select_modes=True,
                        min_modes=min_modes,
                        max_modes=max_modes,
                    )
                    info = results['projection_info']
                    errs = results['reconstruction_error']
                    scan_summary['energy_scan'].append({
                        'energy_threshold': float(thr),
                        'latent_dim': info['latent_dim'],
                        'input_dim': info['input_dim'],
                        'output_dim': info['output_dim'],
                        'input_compression_ratio': info['input_dim'] / max(1, info['latent_dim']),
                        'output_compression_ratio': info['output_dim'] / max(1, info['latent_dim']),
                        'input_relative_error': float(errs['input_relative_error']),
                        'output_relative_error': float(errs['output_relative_error'])
                    })
                except Exception as e:
                    logger.error(f"energy_threshold={thr} 扫描失败: {e}")
        
        # 排序结果
        scan_summary['n_modes_scan'].sort(key=lambda x: x['n_modes'])
        scan_summary['energy_scan'].sort(key=lambda x: x['energy_threshold'])
        
        # 保存 JSON
        json_path = self.output_dir / 'param_scan_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scan_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"参数扫描结果(JSON)已保存到: {json_path}")
        
        # 保存 CSV（各一份）
        import csv
        if scan_summary['n_modes_scan']:
            csv_path = self.output_dir / 'param_scan_n_modes.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(scan_summary['n_modes_scan'][0].keys()))
                writer.writeheader()
                writer.writerows(scan_summary['n_modes_scan'])
            logger.info(f"n_modes 扫描结果(CSV)已保存到: {csv_path}")
        if scan_summary['energy_scan']:
            csv_path = self.output_dir / 'param_scan_energy.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(scan_summary['energy_scan'][0].keys()))
                writer.writeheader()
                writer.writerows(scan_summary['energy_scan'])
            logger.info(f"能量阈值 扫描结果(CSV)已保存到: {csv_path}")
        
        return scan_summary

    # 新增：扫描结果可视化
    def visualize_param_scans(self, scan_summary: Dict):
        """根据扫描结果生成可视化图表"""
        created = []
        # 1) n_modes 扫描图
        nscan = scan_summary.get('n_modes_scan') or []
        if nscan:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            xs = [d['n_modes'] for d in nscan]
            yi = [d['input_relative_error'] for d in nscan]
            yo = [d['output_relative_error'] for d in nscan]
            ax.plot(xs, yi, 'o-', label='输入相对误差')
            ax.plot(xs, yo, 's--', label='输出相对误差')
            ax.set_xlabel('n_modes')
            ax.set_ylabel('相对误差')
            ax.set_title('参数扫描：不同 n_modes 的重建相对误差')
            ax.grid(True, alpha=0.3)
            ax.legend()
            out_png = self.output_dir / 'svd_scan_n_modes.png'
            out_svg = self.output_dir / 'svd_scan_n_modes.svg'
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.savefig(out_svg, format='svg', bbox_inches='tight')
            plt.close()
            logger.info(f"n_modes 扫描图已保存到: {out_png}")
            created.append(out_png)
        
        # 2) 能量阈值扫描图
        escan = scan_summary.get('energy_scan') or []
        if escan:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
            xs = [d['energy_threshold'] for d in escan]
            yi = [d['input_relative_error'] for d in escan]
            yo = [d['output_relative_error'] for d in escan]
            ld = [d['latent_dim'] for d in escan]
            ax1.plot(xs, yi, 'o-', color='tab:blue', label='输入相对误差')
            ax1.plot(xs, yo, 's--', color='tab:red', label='输出相对误差')
            ax1.set_xlabel('能量阈值')
            ax1.set_ylabel('相对误差')
            ax1.grid(True, alpha=0.3)
            ax2 = ax1.twinx()
            ax2.plot(xs, ld, 'd-.', color='tab:green', label='潜在维度')
            ax2.set_ylabel('潜在维度')
            # 合并图例
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='best')
            ax1.set_title('参数扫描：不同能量阈值的误差与潜在维度')
            out_png = self.output_dir / 'svd_scan_energy.png'
            out_svg = self.output_dir / 'svd_scan_energy.svg'
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.savefig(out_svg, format='svg', bbox_inches='tight')
            plt.close()
            logger.info(f"能量阈值扫描图已保存到: {out_png}")
            created.append(out_png)
        
        return created
    
    def visualize_modality_analysis(self, analysis_results: Dict):
        """可视化模态分析结果"""
        logger.info("生成模态分析可视化...")
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HDF5数据集模态分析结果', fontsize=16, fontweight='bold')
        
        # 1. 奇异值分布
        ax = axes[0, 0]
        n_show = min(50, len(analysis_results['input_singular_values']))
        ax.semilogy(range(1, n_show+1), analysis_results['input_singular_values'][:n_show], 
                   'b-o', label='输入奇异值', markersize=4)
        ax.semilogy(range(1, n_show+1), analysis_results['output_singular_values'][:n_show], 
                   'r-s', label='输出奇异值', markersize=4)
        ax.set_xlabel('模态索引')
        ax.set_ylabel('奇异值 (对数尺度)')
        ax.set_title('奇异值衰减曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 能量比例
        ax = axes[0, 1]
        # 使用双柱并排显示，避免遮挡
        n_show = min(30, len(analysis_results['input_energy_ratios']), len(analysis_results['output_energy_ratios']))
        indices = np.arange(1, n_show + 1)
        width = 0.4
        ax.bar(indices - width/2, analysis_results['input_energy_ratios'][:n_show], 
               width=width, alpha=0.8, label='输入能量比例', color='blue')
        ax.bar(indices + width/2, analysis_results['output_energy_ratios'][:n_show], 
               width=width, alpha=0.8, label='输出能量比例', color='red')
        ax.set_xlabel('模态索引')
        ax.set_ylabel('能量比例')
        ax.set_title('各模态能量贡献')
        ax.set_xticks(indices)
        ax.set_xticklabels([str(i) for i in indices])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 累积能量
        ax = axes[0, 2]
        ax.plot(range(1, len(analysis_results['input_cumulative_energy'])+1), 
                analysis_results['input_cumulative_energy'], 'b-', label='输入累积能量', linewidth=2)
        ax.plot(range(1, len(analysis_results['output_cumulative_energy'])+1), 
                analysis_results['output_cumulative_energy'], 'r-', label='输出累积能量', linewidth=2)
        ax.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='90%能量线')
        ax.axhline(y=0.95, color='k', linestyle=':', alpha=0.5, label='95%能量线')
        ax.axhline(y=0.99, color='k', linestyle='-.', alpha=0.5, label='99%能量线')
        ax.set_xlabel('模态数量')
        ax.set_ylabel('累积能量比例')
        ax.set_title('累积能量保持曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # 4. 重建误差
        ax = axes[1, 0]
        error_names = ['输入MSE', '输出MSE', '输入相对误差', '输出相对误差']
        error_values = [
            analysis_results['reconstruction_error']['input_mse'],
            analysis_results['reconstruction_error']['output_mse'],
            analysis_results['reconstruction_error']['input_relative_error'],
            analysis_results['reconstruction_error']['output_relative_error']
        ]
        bars = ax.bar(error_names, error_values, color=['blue', 'red', 'lightblue', 'lightcoral'])
        ax.set_ylabel('误差值')
        ax.set_title('重建误差统计')
        ax.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, error_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 5. 模态有效性分析
        ax = axes[1, 1]
        latent_dim = analysis_results['projection_info']['latent_dim']
        input_dim = analysis_results['projection_info']['input_dim']
        output_dim = analysis_results['projection_info']['output_dim']
        
        compression_ratios = [
            input_dim / latent_dim,
            output_dim / latent_dim
        ]
        
        ax.bar(['输入压缩比', '输出压缩比'], compression_ratios, 
               color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('压缩比')
        ax.set_title('维度压缩效率')
        
        # 添加压缩比数值标签
        for i, ratio in enumerate(compression_ratios):
            ax.text(i, ratio + ratio*0.01, f'{ratio:.1f}x', 
                   ha='center', va='bottom', fontweight='bold')
        
        # 6. 配置信息展示
        ax = axes[1, 2]
        ax.axis('off')
        
        config_text = f"""配置参数:
• 潜在维度: {latent_dim}
• 输入维度: {input_dim:,}
• 输出维度: {output_dim:,}
• 能量阈值: {analysis_results['config']['energy_threshold']}
• 标准化: {analysis_results['config']['standardize_data']}
• 自动选择模态: {analysis_results['config']['auto_select_modes']}

性能指标:
• 输入重建相对误差: {analysis_results['reconstruction_error']['input_relative_error']:.4f}
• 输出重建相对误差: {analysis_results['reconstruction_error']['output_relative_error']:.4f}
• 输入压缩比: {compression_ratios[0]:.1f}x
• 输出压缩比: {compression_ratios[1]:.1f}x"""
        
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # 调整布局并为总标题留出空间
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        
        # 保存图像
        output_path = self.output_dir / 'modality_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"模态分析图像已保存到: {output_path}")
        
        # 额外导出矢量图，避免字体渲染差异
        svg_output_path = self.output_dir / 'modality_analysis.svg'
        plt.savefig(svg_output_path, format='svg', bbox_inches='tight')
        logger.info(f"模态分析SVG已保存到: {svg_output_path}")
        
        plt.close()
        
        return output_path
    
    def generate_analysis_report(self, analysis_results: Dict, dataset_info: Dict):
        """生成分析报告"""
        logger.info("生成模态分析报告...")
        
        # 只保留可序列化的元信息，避免将原始numpy数据写入JSON
        dataset_meta = {
            'file_path': dataset_info.get('file_path'),
            'dataset_info': dataset_info.get('dataset_info', {})
        }
        
        report = {
            'dataset_info': dataset_meta,
            'modality_analysis': analysis_results,
            'summary': {
                'effective_modes': analysis_results['projection_info']['latent_dim'],
                'input_compression_ratio': analysis_results['projection_info']['input_dim'] / analysis_results['projection_info']['latent_dim'],
                'output_compression_ratio': analysis_results['projection_info']['output_dim'] / analysis_results['projection_info']['latent_dim'],
                'input_energy_90': next((i+1 for i, energy in enumerate(analysis_results['input_cumulative_energy']) if energy >= 0.9), len(analysis_results['input_cumulative_energy'])),
                'output_energy_90': next((i+1 for i, energy in enumerate(analysis_results['output_cumulative_energy']) if energy >= 0.9), len(analysis_results['output_cumulative_energy'])),
                'input_energy_95': next((i+1 for i, energy in enumerate(analysis_results['input_cumulative_energy']) if energy >= 0.95), len(analysis_results['input_cumulative_energy'])),
                'output_energy_95': next((i+1 for i, energy in enumerate(analysis_results['output_cumulative_energy']) if energy >= 0.95), len(analysis_results['output_cumulative_energy']))
            }
        }
        
        # 保存JSON报告
        report_path = self.output_dir / 'modality_analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模态分析报告已保存到: {report_path}")
        
        # 生成markdown摘要
        markdown_path = self.output_dir / 'modality_analysis_summary.md'
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# HDF5数据集模态分析报告\n\n")
            f.write("## 数据集信息\n")
            f.write(f"- 文件路径: `{dataset_meta['file_path']}`\n")
            for key, info in dataset_meta['dataset_info'].items():
                f.write(f"- {key}: 形状={info['shape']}, 大小={info['size_mb']:.2f}MB\n")
            
            f.write("\n## 模态分析结果\n")
            f.write(f"- **有效模态数**: {report['summary']['effective_modes']}\n")
            f.write(f"- **输入维度压缩比**: {report['summary']['input_compression_ratio']:.1f}x\n")
            f.write(f"- **输出维度压缩比**: {report['summary']['output_compression_ratio']:.1f}x\n")
            f.write(f"- **输入重建相对误差**: {analysis_results['reconstruction_error']['input_relative_error']:.4f}\n")
            f.write(f"- **输出重建相对误差**: {analysis_results['reconstruction_error']['output_relative_error']:.4f}\n")
            
            f.write("\n## 能量保持分析\n")
            f.write(f"- **输入90%能量所需模态数**: {report['summary']['input_energy_90']}\n")
            f.write(f"- **输入95%能量所需模态数**: {report['summary']['input_energy_95']}\n")
            f.write(f"- **输出90%能量所需模态数**: {report['summary']['output_energy_90']}\n")
            f.write(f"- **输出95%能量所需模态数**: {report['summary']['output_energy_95']}\n")
            
            f.write("\n## 建议\n")
            if report['summary']['effective_modes'] < 32:
                f.write("- 数据具有很强的低维结构，建议使用较小的潜在维度进行训练\n")
            elif report['summary']['effective_modes'] > 128:
                f.write("- 数据复杂度较高，可能需要更多的模态来保持足够的信息\n")
            else:
                f.write("- 数据复杂度适中，当前的模态数设置合理\n")
                
            if min(analysis_results['reconstruction_error']['input_relative_error'], 
                   analysis_results['reconstruction_error']['output_relative_error']) < 0.1:
                f.write("- 重建质量很好，SVD投影能够很好地保持数据信息\n")
            else:
                f.write("- 重建误差较大，可能需要增加模态数或检查数据预处理\n")
        
        logger.info(f"模态分析摘要已保存到: {markdown_path}")
        
        return report_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HDF5数据集模态分析')
    parser.add_argument('--data-path', type=str, 
                       default=r'x:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5',
                       help='HDF5数据文件路径')
    parser.add_argument('--n-modes', type=int, default=64, help='SVD模态数量')
    parser.add_argument('--max-samples', type=int, default=1000, help='最大样本数量')
    parser.add_argument('--output-dir', type=str, default='modality_analysis_results', help='输出目录')
    # 新增参数
    parser.add_argument('--downsample-factor', type=int, default=2, help='空间分块下采样因子（时间维为1或2D数据时生效），输入维度=因子^2')
    parser.add_argument('--energy-threshold', type=float, default=0.99, help='能量阈值，用于自动选择模态数')
    parser.add_argument('--disable-auto-select', action='store_true', help='禁用自动选择模态数，强制使用n-modes')
    parser.add_argument('--min-modes', type=int, default=8, help='最小模态数约束')
    parser.add_argument('--max-modes', type=int, default=0, help='最大模态数约束（0或负数表示自动根据数据确定）')
    # 参数扫描：逗号分隔列表
    parser.add_argument('--scan-n-modes', type=str, default='', help='逗号分隔的模态数列表，触发扫描（禁用自动选择），例如: 8,16,32,64')
    parser.add_argument('--scan-energy', type=str, default='', help='逗号分隔的能量阈值列表，触发扫描（启用自动选择），例如: 0.9,0.95,0.99')
    
    args = parser.parse_args()
    
    # 设置路径
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    
    logger.info("=== HDF5数据集模态分析开始 ===")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"分析模态数: {args.n_modes}")
    
    # 创建分析器
    analyzer = HDF5ModalityAnalyzer(output_dir)
    
    try:
        # 1. 加载数据集
        dataset_info = analyzer.load_hdf5_dataset(data_path)
        if dataset_info is None:
            logger.error("数据集加载失败")
            return
        
        # 2. 预处理数据
        input_data, output_data = analyzer.preprocess_data(dataset_info['data'], args.max_samples, args.downsample_factor)
        if input_data is None:
            logger.error("数据预处理失败")
            return
        
        # 3. 进行SVD模态分析
        analysis_results, projector = analyzer.perform_svd_analysis(
            input_data, output_data,
            n_modes=args.n_modes,
            energy_threshold=args.energy_threshold,
            auto_select_modes=(not args.disable_auto_select),
            min_modes=args.min_modes,
            max_modes=(args.max_modes if args.max_modes > 0 else None)
        )
        
        # 4. 生成可视化
        analyzer.visualize_modality_analysis(analysis_results)
        
        # 5. 生成分析报告
        analyzer.generate_analysis_report(analysis_results, dataset_info)
        
        # 6. （可选）参数扫描
        scan_n_modes = [int(x.strip()) for x in args.scan_n_modes.split(',') if x.strip()] if args.scan_n_modes else []
        scan_energy = [float(x.strip()) for x in args.scan_energy.split(',') if x.strip()] if args.scan_energy else []
        if scan_n_modes or scan_energy:
            logger.info(f"触发参数扫描: scan_n_modes={scan_n_modes}, scan_energy={scan_energy}")
            scan_summary = analyzer.scan_svd_parameters(
                input_data, output_data,
                n_modes_list=scan_n_modes if scan_n_modes else None,
                energy_threshold_list=scan_energy if scan_energy else None,
                min_modes=args.min_modes,
                max_modes=(args.max_modes if args.max_modes > 0 else None)
            )
            created = analyzer.visualize_param_scans(scan_summary)
            logger.info(f"参数扫描完成。产物：JSON/CSV与图像，输出目录: {output_dir}")
            for p in created:
                logger.info(f"  - {p}")
        
        logger.info("=== 模态分析完成 ===")
        logger.info(f"结果保存在: {output_dir}")
        logger.info(f"主要发现:")
        logger.info(f"  - 有效模态数: {analysis_results['projection_info']['latent_dim']}")
        logger.info(f"  - 输入压缩比: {analysis_results['projection_info']['input_dim'] / analysis_results['projection_info']['latent_dim']:.1f}x")
        logger.info(f"  - 输出压缩比: {analysis_results['projection_info']['output_dim'] / analysis_results['projection_info']['latent_dim']:.1f}x")
        logger.info(f"  - 重建相对误差: 输入={analysis_results['reconstruction_error']['input_relative_error']:.4f}, 输出={analysis_results['reconstruction_error']['output_relative_error']:.4f}")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        raise

# 修复中文字体设置（加强版）
def configure_chinese_font_enhanced():
    """加强版中文字体配置：注册系统字体，选择可用字体，过滤警告"""
    try:
        import warnings
        
        # 过滤字体相关警告
        warnings.filterwarnings("ignore", message=r".*Glyph.*missing from font.*")
        warnings.filterwarnings("ignore", message=r".*does not have a glyph.*")
        warnings.filterwarnings("ignore", message=r".*substituting with a dummy symbol.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        
        # 在Windows上注册常见中文字体文件
        possible_font_files = [
            r"C:\\Windows\\Fonts\\msyh.ttc",      # 微软雅黑
            r"C:\\Windows\\Fonts\\msyh.ttf",
            r"C:\\Windows\\Fonts\\simhei.ttf",   # 黑体
            r"C:\\Windows\\Fonts\\simsun.ttc",   # 宋体
            r"C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc",
            r"C:\\Windows\\Fonts\\SourceHanSansSC-Regular.otf",
        ]
        for fp in possible_font_files:
            if os.path.exists(fp):
                try:
                    fm.fontManager.addfont(fp)
                except Exception:
                    pass
        
        # 候选字体（按优先级）
        candidate_fonts = [
            "Microsoft YaHei",
            "Microsoft YaHei UI",
            "DengXian",
            "NSimSun",
            "SimSun",                # 宋体
            "SimHei",
            "KaiTi",
            "FangSong",
            "PingFang SC",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Arial Unicode MS",      # 对Unicode支持较好
            "WenQuanYi Zen Hei",
            "HarmonyOS Sans SC",
            "MiSans",
            "DejaVu Sans",
        ]
        
        # 选择第一个可用的字体
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
            selected = "SimSun"  # 退回到宋体（通常有更好的Unicode支持）
        
        # 应用字体配置（更强覆盖与兼容）
        matplotlib.rcParams['font.family'] = [selected, 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        # 统一使用全局 sitecustomize.py 的中文字体和负号设置
        # matplotlib.rcParams['font.sans-serif'] = [selected] + [f for f in candidate_fonts if f != selected]
        # matplotlib.rcParams['axes.unicode_minus'] = False
        # 数学文本与符号（如希腊字母）使用 STIX 提供更完整支持
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        # 特别设置：确保SVG导出时文字转换为路径，避免跨平台字体问题（已在全局 sitecustomize 设置）
        # matplotlib.rcParams['svg.fonttype'] = 'path'
        
        print(f"已配置字体: {selected}")
    except Exception as e:
        print(f"字体配置失败: {e}")

# 在模块加载时立即配置字体
configure_chinese_font_enhanced()
sns.set_style("whitegrid")

if __name__ == "__main__":
    main()