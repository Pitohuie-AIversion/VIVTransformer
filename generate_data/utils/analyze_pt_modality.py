#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch .pt 文件模态分析脚本
专门针对 merged_all_pressures_separated_normalized.pt 等 .pt 格式文件进行 SVD 模态分解

功能:
1. 加载 .pt 文件中的 torch tensor 数据
2. 自动识别文件结构（支持 Dict 和单 Tensor）
3. 对数据进行 SVD 分解和模态分析
4. 可视化模态结构和能量分布
5. 生成分析报告（支持中文字体）

文件结构支持:
- Dict 格式: {"reynolds_data": {...}, "reynolds_keys": [...]}
- Tensor 格式: 直接的 torch.Tensor

Usage:
    python analyze_pt_modality.py --data-path path/to/data.pt [--n-modes 64] [--output-dir results]
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
import torch
import json
from typing import Dict, List, Tuple, Optional, Union

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
    """配置中文字体支持"""
    global CHINESE_FONT_NAME
    try:
        import matplotlib.font_manager as fm
        import matplotlib as mpl

        # 候选中文字体列表
        candidate_fonts = [
            "Source Han Sans SC", "Noto Sans SC", "Noto Sans CJK SC",
            "Microsoft YaHei", "SimHei", "STSong", "STKaiti",
            "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "DejaVu Sans"
        ]
        
        # 查找系统可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 选择第一个可用的中文字体
        selected = None
        for font in candidate_fonts:
            if font in available_fonts:
                selected = font
                break
        
        if not selected:
            selected = "DejaVu Sans"  # 降级到默认字体
            logger.warning("未找到中文字体，使用 DejaVu Sans")
        else:
            logger.info(f"选择中文字体: {selected}")
        
        # 设置全局字体
        CHINESE_FONT_NAME = selected
        if selected in candidate_fonts:
            mpl.rcParams["font.family"] = selected
            mpl.rcParams["font.sans-serif"] = [selected] + [f for f in candidate_fonts if f != selected]
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["svg.fonttype"] = "none"  # SVG 引用字体
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


class PtModalityAnalyzer:
    """PyTorch .pt 文件模态分析器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_pt_dataset(self, data_path: Path) -> Optional[Dict]:
        """加载 .pt 文件数据集"""
        logger.info(f"正在加载 .pt 数据集: {data_path}")
        
        if not data_path.exists():
            logger.error(f"数据文件不存在: {data_path}")
            return None
            
        try:
            # 加载 .pt 文件
            data = torch.load(data_path, map_location='cpu')
            logger.info(f"成功加载 .pt 文件，类型: {type(data)}")
            
            # 分析数据结构
            if isinstance(data, dict):
                logger.info("检测到 Dict 格式数据")
                return self._analyze_dict_structure(data, data_path)
            elif isinstance(data, torch.Tensor):
                logger.info("检测到 Tensor 格式数据")
                return self._analyze_tensor_structure(data, data_path)
            else:
                logger.error(f"不支持的数据类型: {type(data)}")
                return None
                
        except Exception as e:
            logger.error(f"加载 .pt 文件时出错: {str(e)}")
            return None
    
    def _analyze_dict_structure(self, data: Dict, data_path: Path) -> Dict:
        """分析 Dict 结构的数据"""
        keys = list(data.keys())
        logger.info(f"Dict 包含的键: {keys}")
        
        dataset_info = {
            'file_path': str(data_path),
            'data_type': 'dict',
            'keys': keys,
            'dict_info': {}
        }
        
        # 分析每个键的内容
        for key in keys:
            value = data[key]
            if isinstance(value, torch.Tensor):
                dataset_info['dict_info'][key] = {
                    'type': 'tensor',
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'size_mb': value.numel() * value.element_size() / (1024**2)
                }
                logger.info(f"  {key}: 形状={value.shape}, 类型={value.dtype}, 大小={dataset_info['dict_info'][key]['size_mb']:.2f}MB")
            elif isinstance(value, dict):
                dataset_info['dict_info'][key] = {
                    'type': 'dict',
                    'sub_keys': list(value.keys()),
                    'size': len(value)
                }
                logger.info(f"  {key}: 字典，包含 {len(value)} 个子键: {list(value.keys())[:5]}...")
            elif isinstance(value, list):
                dataset_info['dict_info'][key] = {
                    'type': 'list',
                    'length': len(value),
                    'sample_items': value[:3] if len(value) > 0 else []
                }
                logger.info(f"  {key}: 列表，长度 {len(value)}")
            else:
                dataset_info['dict_info'][key] = {
                    'type': str(type(value)),
                    'value': str(value)[:100]
                }
                logger.info(f"  {key}: {type(value)}")
        
        # 提取主要数据进行分析
        main_data = self._extract_main_data(data)
        if main_data is not None:
            dataset_info['main_data'] = main_data
            
        return dataset_info
    
    def _analyze_tensor_structure(self, data: torch.Tensor, data_path: Path) -> Dict:
        """分析 Tensor 结构的数据"""
        dataset_info = {
            'file_path': str(data_path),
            'data_type': 'tensor',
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'size_mb': data.numel() * data.element_size() / (1024**2),
            'main_data': data.numpy().astype(np.float32)
        }
        
        logger.info(f"Tensor 形状: {data.shape}, 类型: {data.dtype}, 大小: {dataset_info['size_mb']:.2f}MB")
        
        return dataset_info
    
    def _extract_main_data(self, data: Dict) -> Optional[np.ndarray]:
        """从 Dict 中提取主要数据用于分析"""
        # 针对 merged_all_pressures_separated_normalized.pt 的结构
        if "reynolds_data" in data and "reynolds_keys" in data:
            reynolds_data = data["reynolds_data"]
            reynolds_keys = data["reynolds_keys"]
            
            logger.info(f"发现雷诺数数据，包含 {len(reynolds_keys)} 个雷诺数: {reynolds_keys}")
            
            # 合并所有雷诺数的压力数据
            all_pressure_data = []
            
            for reynolds_key in reynolds_keys:
                if reynolds_key in reynolds_data:
                    reynolds_info = reynolds_data[reynolds_key]
                    if "pressure" in reynolds_info:
                        pressure_tensor = reynolds_info["pressure"]
                        if isinstance(pressure_tensor, torch.Tensor):
                            # 转换为 numpy 并添加到列表
                            pressure_np = pressure_tensor.numpy().astype(np.float32)
                            all_pressure_data.append(pressure_np)
                            logger.info(f"雷诺数 {reynolds_key}: pressure 形状 {pressure_np.shape}")
            
            if all_pressure_data:
                # 沿第一个维度（样本维度）合并
                combined_data = np.concatenate(all_pressure_data, axis=0)
                logger.info(f"合并后的数据形状: {combined_data.shape}")
                return combined_data
        
        # 通用提取：寻找最大的 tensor
        max_tensor = None
        max_size = 0
        
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.numel() > max_size:
                max_tensor = value
                max_size = value.numel()
        
        if max_tensor is not None:
            logger.info(f"使用最大的 tensor 进行分析，形状: {max_tensor.shape}")
            return max_tensor.numpy().astype(np.float32)
        
        logger.warning("未找到合适的数据进行分析")
        return None
    
    def preprocess_data(self, data: np.ndarray, max_samples: int = 1000) -> np.ndarray:
        """预处理数据 - 将多维数据展平为2D矩阵"""
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
                
        elif len(data.shape) == 3:  # [samples, height, width] 或 [time, height, width]
            n_samples, H, W = data.shape
            logger.info(f"3D数据: {n_samples}样本, {H}x{W}空间")
            
        elif len(data.shape) == 2:  # [samples, features]
            logger.info(f"2D数据: {data.shape[0]}样本, {data.shape[1]}特征")
            # 已经是展平的格式，直接使用
            if data.shape[0] > max_samples:
                indices = np.random.choice(data.shape[0], max_samples, replace=False)
                data = data[indices]
                logger.info(f"随机选择 {max_samples} 个样本")
            return data
            
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
    
    def perform_svd_analysis(self, data: np.ndarray, n_modes: int = 64, energy_threshold: float = 0.99) -> Dict:
        """对数据进行 SVD 模态分解"""
        logger.info(f"开始 SVD 分析，数据形状: {data.shape}")
        
        n_samples, output_dim = data.shape
        
        # 标准化数据
        data_mean = np.mean(data, axis=0)
        data_std = data - data_mean
        
        # 执行 SVD
        logger.info("执行 SVD 分解...")
        U, S, Vt = np.linalg.svd(data_std, full_matrices=False)
        
        # 限制模态数量
        effective_rank = min(len(S), n_modes, n_samples, output_dim)
        S = S[:effective_rank]
        U = U[:, :effective_rank]
        Vt = Vt[:effective_rank, :]
        
        logger.info(f"SVD 完成，有效模态数: {effective_rank}")
        
        # 计算能量
        total_energy = np.sum(S**2)
        cumulative_energy = np.cumsum(S**2) / total_energy
        
        # 根据能量阈值确定有效模态数
        auto_modes = np.argmax(cumulative_energy >= energy_threshold) + 1
        effective_modes = min(auto_modes, effective_rank)
        
        logger.info(f"能量阈值 {energy_threshold} 对应的模态数: {auto_modes}")
        logger.info(f"实际使用的有效模态数: {effective_modes}")
        
        # 重建数据评估误差
        U_reduced = U[:, :effective_modes]
        S_reduced = S[:effective_modes]
        Vt_reduced = Vt[:effective_modes, :]
        
        data_reconstructed = U_reduced @ np.diag(S_reduced) @ Vt_reduced + data_mean
        
        mse = np.mean((data_std - (data_reconstructed - data_mean))**2)
        relative_error = np.sqrt(mse) / (np.std(data_std) + 1e-8)
        
        logger.info(f"重建误差: MSE={mse:.6f}, 相对误差={relative_error:.6f}")
        
        # 整理结果
        analysis_results = {
            'singular_values': S,
            'energy_ratios': (S**2) / total_energy,
            'cumulative_energy': cumulative_energy,
            'effective_modes': effective_modes,
            'output_dim': output_dim,
            'n_samples': n_samples,
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
        """可视化分析结果"""
        logger.info("生成可视化图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # 设置整体标题
        fig.suptitle('PyTorch .pt 文件数据模态分析结果', fontsize=16, fontweight='bold')
        
        S = analysis_results['singular_values']
        energy_ratios = analysis_results['energy_ratios']
        cumulative_energy = analysis_results['cumulative_energy']
        effective_modes = analysis_results['effective_modes']
        
        # 1. 奇异值分布
        ax = axes[0]
        ax.semilogy(S, 'b-', linewidth=2, marker='o', markersize=4)
        ax.axvline(x=effective_modes-1, color='red', linestyle='--', 
                  label=f'有效模态数: {effective_modes}')
        ax.set_xlabel('模态索引')
        ax.set_ylabel('奇异值 (对数尺度)')
        ax.set_title('奇异值分布')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. 能量分布
        ax = axes[1]
        ax.bar(range(min(20, len(energy_ratios))), energy_ratios[:20], alpha=0.7, color='green')
        ax.set_xlabel('模态索引')
        ax.set_ylabel('能量比例')
        ax.set_title('前20个模态的能量分布')
        ax.grid(True, alpha=0.3)
        
        # 3. 累积能量
        ax = axes[2]
        ax.plot(cumulative_energy, 'r-', linewidth=2, marker='s', markersize=3)
        ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%能量')
        ax.axhline(y=0.95, color='purple', linestyle='--', alpha=0.7, label='95%能量')
        ax.axhline(y=0.99, color='brown', linestyle='--', alpha=0.7, label='99%能量')
        ax.axvline(x=effective_modes-1, color='red', linestyle='--', 
                  label=f'有效模态数: {effective_modes}')
        ax.set_xlabel('模态索引')
        ax.set_ylabel('累积能量比例')
        ax.set_title('累积能量保持')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. 统计信息
        ax = axes[3]
        ax.axis('off')
        
        # 计算关键统计量
        energy_90 = np.argmax(cumulative_energy >= 0.9) + 1
        energy_95 = np.argmax(cumulative_energy >= 0.95) + 1
        energy_99 = np.argmax(cumulative_energy >= 0.99) + 1
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
        
        # 强制设置中文字体
        try:
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
        except Exception as e:
            logger.warning(f"设置中文字体时出错: {e}")
        
        # 保存图像（同时生成文本版和路径版 SVG）
        png_path = self.output_dir / 'pt_modality_analysis.png'
        svg_path = self.output_dir / 'pt_modality_analysis.svg'
        svg_path_path = self.output_dir / 'pt_modality_analysis_path.svg'
        
        # 保存 PNG
        plt.savefig(png_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # 保存文本版 SVG (svg.fonttype='none')
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(svg_path, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        # 保存路径版 SVG (svg.fonttype='path')
        plt.rcParams["svg.fonttype"] = "path"
        plt.savefig(svg_path_path, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        plt.close()
        
        logger.info(f"可视化图表已保存:")
        logger.info(f"  PNG: {png_path}")
        logger.info(f"  SVG (文本版): {svg_path}")
        logger.info(f"  SVG (路径版): {svg_path_path}")
    
    def generate_report(self, analysis_results: Dict, dataset_info: Dict):
        """生成分析报告"""
        logger.info("生成模态分析报告...")
        
        # 计算关键统计量
        effective_modes = analysis_results['effective_modes']
        energy_90 = np.argmax(analysis_results['cumulative_energy'] >= 0.9) + 1
        energy_95 = np.argmax(analysis_results['cumulative_energy'] >= 0.95) + 1
        energy_99 = np.argmax(analysis_results['cumulative_energy'] >= 0.99) + 1
        compression_ratio = analysis_results['output_dim'] / effective_modes
        
        report = {
            'dataset_info': dataset_info,
            'analysis_type': 'pt_modality',
            'modality_analysis': analysis_results,
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
            import torch as _torch
            if isinstance(o, _np.ndarray):
                return o.tolist()
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, _torch.Tensor):
                return o.detach().cpu().numpy().tolist()
            if isinstance(o, dict):
                return {k: _to_json_safe(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_to_json_safe(x) for x in o]
            return o

        # 保存JSON报告
        report_path = self.output_dir / 'pt_modality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(_to_json_safe(report), f, indent=2, ensure_ascii=False)
        
        logger.info(f"模态分析报告已保存到: {report_path}")
        
        # 生成markdown摘要
        markdown_path = self.output_dir / 'pt_modality_summary.md'
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# PyTorch .pt 文件模态分析报告\n\n")
            
            f.write("## 数据集信息\n")
            f.write(f"- 文件路径: `{dataset_info['file_path']}`\n")
            f.write(f"- 数据类型: {dataset_info['data_type']}\n")
            
            if dataset_info['data_type'] == 'dict':
                f.write(f"- 包含键: {dataset_info['keys']}\n")
                for key, info in dataset_info['dict_info'].items():
                    if info.get('type') == 'tensor':
                        f.write(f"- {key}: 形状={info['shape']}, 大小={info['size_mb']:.2f}MB\n")
            elif dataset_info['data_type'] == 'tensor':
                f.write(f"- 张量形状: {dataset_info['shape']}\n")
                f.write(f"- 大小: {dataset_info['size_mb']:.2f}MB\n")
            
            f.write("\n## 模态分析结果\n")
            f.write(f"- **分析类型**: PyTorch .pt 文件模态分析\n")
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
        
        logger.info(f"模态分析摘要已保存到: {markdown_path}")
        return report_path
    
    def perform_default_analysis(self, dataset_info: Dict, args):
        """执行默认分析（原有逻辑）"""
        logger.info("执行默认模态分析...")
        
        # 2. 提取主要数据用于分析
        main_data = dataset_info.get('main_data')
        if main_data is None:
            logger.error("未找到可分析的数据")
            return
        
        # 3. 预处理数据
        processed_data = self.preprocess_data(main_data, args.max_samples)
        if processed_data is None:
            logger.error("数据预处理失败")
            return
        
        # 4. 进行 SVD 分析
        analysis_results = self.perform_svd_analysis(processed_data, args.n_modes, args.energy_threshold)
        
        # 5. 可视化结果
        self.visualize_analysis(analysis_results)
        
        # 6. 生成报告
        self.generate_report(analysis_results, dataset_info)
    
    def perform_comprehensive_analysis(self, dataset_info: Dict, args):
        """执行全量分析"""
        logger.info("执行全量模态分析...")
        
        # 直接加载原始数据进行详细分析（不依赖预处理提取的main_data）
        data_path = Path(dataset_info['file_path'])
        original_data = torch.load(data_path, map_location='cpu')
        
        if not isinstance(original_data, dict) or 'reynolds_data' not in original_data:
            logger.warning("数据格式不支持全量分析，回退到默认分析")
            self.perform_default_analysis(dataset_info, args)
            return
        
        reynolds_data = original_data['reynolds_data']
        reynolds_keys = original_data.get('reynolds_keys', list(reynolds_data.keys()))
        
        comprehensive_results = {}
        
        # 1. 按雷诺数分析
        if args.per_reynolds:
            logger.info("按雷诺数分别分析...")
            comprehensive_results['per_reynolds'] = {}
            
            for reynolds_key in reynolds_keys:
                if reynolds_key in reynolds_data:
                    logger.info(f"分析雷诺数: {reynolds_key}")
                    reynolds_info = reynolds_data[reynolds_key]
                    
                    # 分析此雷诺数下的所有变量
                    if args.analyze_variables:
                        for var_name in ['pressure', 'in_pressure']:
                            if var_name in reynolds_info:
                                result = self._analyze_single_variable(
                                    reynolds_info[var_name], 
                                    f"Re_{reynolds_key}_{var_name}",
                                    args
                                )
                                if result:
                                    comprehensive_results['per_reynolds'][f"{reynolds_key}_{var_name}"] = result
                    else:
                        # 只分析pressure
                        if 'pressure' in reynolds_info:
                            result = self._analyze_single_variable(
                                reynolds_info['pressure'], 
                                f"Re_{reynolds_key}_pressure",
                                args
                            )
                            if result:
                                comprehensive_results['per_reynolds'][reynolds_key] = result
        
        # 2. 按变量分析（合并所有雷诺数）
        if args.analyze_variables and not args.per_reynolds:
            logger.info("按变量分析（合并所有雷诺数）...")
            comprehensive_results['per_variable'] = {}
            
            for var_name in ['pressure', 'in_pressure']:
                # 收集所有雷诺数的该变量数据
                var_data_list = []
                for reynolds_key in reynolds_keys:
                    if reynolds_key in reynolds_data and var_name in reynolds_data[reynolds_key]:
                        var_tensor = reynolds_data[reynolds_key][var_name]
                        var_data_list.append(var_tensor.numpy().astype(np.float32))
                
                if var_data_list:
                    combined_var_data = np.concatenate(var_data_list, axis=0)
                    result = self._analyze_processed_data(combined_var_data, f"all_Re_{var_name}", args)
                    if result:
                        comprehensive_results['per_variable'][var_name] = result
        
        # 3. 时间步分析
        if args.analyze_time_steps:
            logger.info("分析时间步维度...")
            comprehensive_results['per_time_step'] = {}
            
            # 使用第一个雷诺数的pressure数据作为示例
            first_reynolds = reynolds_keys[0]
            if first_reynolds in reynolds_data and 'pressure' in reynolds_data[first_reynolds]:
                pressure_data = reynolds_data[first_reynolds]['pressure'].numpy().astype(np.float32)
                
                if len(pressure_data.shape) >= 2:  # 确保有时间维度
                    time_steps = pressure_data.shape[1] if len(pressure_data.shape) > 2 else 1
                    
                    for t in range(min(time_steps, 5)):  # 最多分析前5个时间步
                        if len(pressure_data.shape) == 4:  # [samples, time, H, W]
                            time_data = pressure_data[:, t, :, :]
                        elif len(pressure_data.shape) == 3:  # [samples, time, features]
                            time_data = pressure_data[:, t, :]
                        else:
                            continue
                        
                        result = self._analyze_processed_data(time_data, f"time_step_{t}", args)
                        if result:
                            comprehensive_results['per_time_step'][f"t_{t}"] = result
        
        # 生成综合报告
        self._generate_comprehensive_report(comprehensive_results, dataset_info, args)
    
    def _analyze_single_variable(self, tensor_data: torch.Tensor, analysis_name: str, args) -> Optional[Dict]:
        """分析单个变量的张量数据"""
        try:
            data_np = tensor_data.numpy().astype(np.float32)
            return self._analyze_processed_data(data_np, analysis_name, args)
        except Exception as e:
            logger.error(f"分析 {analysis_name} 时出错: {e}")
            return None
    
    def _analyze_processed_data(self, data: np.ndarray, analysis_name: str, args) -> Optional[Dict]:
        """分析预处理后的数据"""
        try:
            logger.info(f"分析 {analysis_name}, 数据形状: {data.shape}")
            
            # 预处理数据
            processed_data = self.preprocess_data(data, args.max_samples)
            if processed_data is None:
                return None
            
            # SVD分析
            analysis_result = self.perform_svd_analysis(processed_data, args.n_modes, args.energy_threshold)
            analysis_result['analysis_name'] = analysis_name
            analysis_result['original_shape'] = data.shape
            
            logger.info(f"{analysis_name} 分析完成: 有效模态数={analysis_result['effective_modes']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"分析 {analysis_name} 时出错: {e}")
            return None
    
    def _generate_comprehensive_report(self, comprehensive_results: Dict, dataset_info: Dict, args):
        """生成全量分析的综合报告"""
        logger.info("生成全量分析综合报告...")
        
        # 保存详细的JSON报告
        def _to_json_safe(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, torch.Tensor):
                return o.detach().cpu().numpy().tolist()
            if isinstance(o, dict):
                return {k: _to_json_safe(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_to_json_safe(x) for x in o]
            return o
        
        comprehensive_report = {
            'dataset_info': dataset_info,
            'analysis_type': 'comprehensive_pt_modality',
            'analysis_config': {
                'per_reynolds': args.per_reynolds,
                'analyze_variables': args.analyze_variables,
                'analyze_time_steps': args.analyze_time_steps,
                'n_modes': args.n_modes,
                'max_samples': args.max_samples,
                'energy_threshold': args.energy_threshold
            },
            'results': comprehensive_results
        }
        
        # 保存JSON报告
        json_path = self.output_dir / 'comprehensive_pt_modality_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_to_json_safe(comprehensive_report), f, indent=2, ensure_ascii=False)
        
        # 生成markdown摘要
        markdown_path = self.output_dir / 'comprehensive_pt_modality_summary.md'
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# PyTorch .pt 文件全量模态分析报告\n\n")
            
            f.write("## 分析配置\n")
            f.write(f"- 按雷诺数分析: {'是' if args.per_reynolds else '否'}\n")
            f.write(f"- 变量分析: {'是' if args.analyze_variables else '否'}\n")
            f.write(f"- 时间步分析: {'是' if args.analyze_time_steps else '否'}\n")
            f.write(f"- 目标模态数: {args.n_modes}\n")
            f.write(f"- 最大样本数: {args.max_samples}\n")
            f.write(f"- 能量阈值: {args.energy_threshold}\n\n")
            
            # 汇总所有分析结果
            all_results = []
            for category, results in comprehensive_results.items():
                if isinstance(results, dict):
                    for name, result in results.items():
                        if isinstance(result, dict) and 'effective_modes' in result:
                            all_results.append({
                                'category': category,
                                'name': name,
                                'effective_modes': result['effective_modes'],
                                'compression_ratio': result['output_dim'] / result['effective_modes'],
                                'reconstruction_error': result['reconstruction_error']['relative_error'],
                                'original_shape': result.get('original_shape', 'N/A')
                            })
            
            if all_results:
                f.write("## 分析结果汇总\n\n")
                f.write("| 分析类别 | 分析对象 | 原始形状 | 有效模态数 | 压缩比 | 重建误差 |\n")
                f.write("|---------|---------|---------|-----------|--------|----------|\n")
                
                for result in all_results:
                    f.write(f"| {result['category']} | {result['name']} | {result['original_shape']} | "
                           f"{result['effective_modes']} | {result['compression_ratio']:.1f}x | "
                           f"{result['reconstruction_error']:.6f} |\n")
                
                f.write("\n## 主要发现\n")
                
                # 统计分析
                effective_modes_list = [r['effective_modes'] for r in all_results]
                compression_ratios = [r['compression_ratio'] for r in all_results]
                errors = [r['reconstruction_error'] for r in all_results]
                
                f.write(f"- 分析了 {len(all_results)} 个不同的数据组合\n")
                f.write(f"- 有效模态数范围: {min(effective_modes_list)} - {max(effective_modes_list)}\n")
                f.write(f"- 平均压缩比: {np.mean(compression_ratios):.1f}x\n")
                f.write(f"- 平均重建误差: {np.mean(errors):.6f}\n")
                
                # 找出最佳和最差的情况
                best_idx = np.argmin(errors)
                worst_idx = np.argmax(errors)
                
                f.write(f"\n### 最佳重建质量\n")
                best = all_results[best_idx]
                f.write(f"- **{best['name']}** (类别: {best['category']})\n")
                f.write(f"- 重建误差: {best['reconstruction_error']:.6f}\n")
                f.write(f"- 有效模态数: {best['effective_modes']}\n")
                f.write(f"- 压缩比: {best['compression_ratio']:.1f}x\n")
                
                f.write(f"\n### 最高压缩比\n")
                best_compression_idx = np.argmax(compression_ratios)
                best_comp = all_results[best_compression_idx]
                f.write(f"- **{best_comp['name']}** (类别: {best_comp['category']})\n")
                f.write(f"- 压缩比: {best_comp['compression_ratio']:.1f}x\n")
                f.write(f"- 有效模态数: {best_comp['effective_modes']}\n")
                f.write(f"- 重建误差: {best_comp['reconstruction_error']:.6f}\n")
            
            else:
                f.write("## 分析结果\n")
                f.write("未找到有效的分析结果。\n")
        
        logger.info(f"全量分析报告已保存:")
        logger.info(f"  JSON详细报告: {json_path}")
        logger.info(f"  Markdown摘要: {markdown_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyTorch .pt 文件模态分析')
    parser.add_argument('--data-path', type=str, 
                       default=r'X:\2025\Graduation_project\simulation_results\merged_all_pressures_separated_normalized.pt',
                       help='.pt 数据文件路径')
    parser.add_argument('--n-modes', type=int, default=64, help='SVD模态数量')
    parser.add_argument('--max-samples', type=int, default=1000, help='最大样本数量')
    parser.add_argument('--output-dir', type=str, default='pt_modality_analysis', help='输出目录')
    parser.add_argument('--energy-threshold', type=float, default=0.99, help='能量阈值，用于自动选择模态数')
    
    # 新增全量分析参数
    parser.add_argument('--per-reynolds', action='store_true', help='按每个雷诺数分别分析')
    parser.add_argument('--analyze-variables', action='store_true', help='分析所有变量（pressure、in_pressure等）')
    parser.add_argument('--analyze-time-steps', action='store_true', help='分析不同时间步')
    parser.add_argument('--analyze-all', action='store_true', help='进行全量分析（包含所有组合）')
    
    args = parser.parse_args()
    
    # 如果设置了 --analyze-all，则启用所有分析选项
    if args.analyze_all:
        args.per_reynolds = True
        args.analyze_variables = True
        args.analyze_time_steps = True
    
    # 设置路径
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    
    logger.info("=== PyTorch .pt 文件模态分析开始 ===")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"分析模态数: {args.n_modes}")
    if args.per_reynolds:
        logger.info("模式: 按雷诺数分别分析")
    if args.analyze_variables:
        logger.info("模式: 分析所有变量")
    if args.analyze_time_steps:
        logger.info("模式: 分析时间步维度")
    
    # 创建分析器
    analyzer = PtModalityAnalyzer(output_dir)
    
    try:
        # 1. 加载数据集
        dataset_info = analyzer.load_pt_dataset(data_path)
        if dataset_info is None:
            logger.error("数据集加载失败")
            return
        
        # 2. 根据参数决定分析方式
        if args.per_reynolds or args.analyze_variables or args.analyze_time_steps:
            # 执行全量分析
            analyzer.perform_comprehensive_analysis(dataset_info, args)
        else:
            # 执行默认分析（原有逻辑）
            analyzer.perform_default_analysis(dataset_info, args)
        
        logger.info("=== 模态分析完成 ===")
        logger.info(f"结果保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()