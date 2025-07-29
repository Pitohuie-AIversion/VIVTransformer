#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流场数据处理和可视化脚本

功能描述:
    1. 从CSV文件中读取流场压力数据
    2. 对数据进行归一化处理
    3. 将数据组织成适合机器学习的格式
    4. 提供数据可视化功能
    5. 支持批量处理和交互式操作

数据格式:
    - 输入: CSV文件，包含x, y坐标和压力值p
    - 输出: PyTorch张量，形状为[reynolds_cases, time_steps, height, width]
    
使用方法:
    1. 直接运行脚本: python script.py
    2. 导入使用: from script import DataProcessor, DataVisualizer
    
作者: [Your Name]
日期: 2025
版本: 2.0
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
class Config:
    """配置类，集中管理所有参数"""
    ROOT_DIRECTORY = r"X:/2025/Graduation_project/simulation_results"
    START_TIME = 3.50
    END_TIME = 7.00
    GRID_SIZE = 200
    INPUT_SIZE = 20
    OUTPUT_FILE = 'merged_all_pressures_separated_normalized.pt'
    CASE_PREFIX = 'case_Re_'
    FILE_PREFIX = 'cylinder_flow_'
    
class DataProcessor:
    """数据处理类，封装所有数据处理逻辑"""
    
    def __init__(self, config: Config, test_mode: bool = False):
        self.config = config
        self.all_data = {}
        self.reynolds_list = []
        self.time_list = []
        self.test_mode = test_mode


    def read_csv_data(self, csv_path: str) -> Optional[Tuple[Dict, str, float]]:
        """
        读取 CSV 文件，返回包含流场数据的字典。
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            包含数据字典、雷诺数和时间步的元组，如果读取失败返回None
        """
        try:
            # 验证文件存在
            if not os.path.exists(csv_path):
                logger.error(f"文件不存在: {csv_path}")
                return None
                
            # 读取 CSV 文件
            data = pd.read_csv(csv_path)
            
            # 检查文件是否为空
            if data.empty:
                logger.error(f"文件 {csv_path} 为空")
                return None
            
            # 清理列名（去除空格）
            data.columns = data.columns.str.strip()
            
            # 验证必要的列是否存在
            required_columns = ['x', 'y', 'p']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"CSV文件 {csv_path} 缺少必要的列: {missing_columns}")
                logger.error(f"文件现有列: {list(data.columns)}")
                return None

            # 检查数据是否包含NaN值
            if data[required_columns].isnull().any().any():
                logger.warning(f"文件 {csv_path} 包含NaN值，将用0填充")

            # 提取坐标和压力值
            sparse_points = data[['x', 'y']].values
            pressure_values = data['p'].values

            # 用0填充NaN值
            pressure_values = np.nan_to_num(pressure_values, nan=0.0)

            # 检查数据是否可以reshape为指定大小
            expected_size = self.config.GRID_SIZE * self.config.GRID_SIZE
            if len(sparse_points) != expected_size:
                logger.warning(f"数据点数量不符合要求: {len(sparse_points)}，需要 {expected_size}")
                # 如果数据点太少，无法处理
                if len(sparse_points) < expected_size:
                    logger.error(f"数据点太少，无法处理")
                    return None

            # 重新排列数据为网格格局
            try:
                pressure_matrix = pressure_values.reshape(self.config.GRID_SIZE, self.config.GRID_SIZE)
            except ValueError as e:
                logger.error(f"无法将压力数据重塑为 {self.config.GRID_SIZE}x{self.config.GRID_SIZE} 网格: {str(e)}")
                return None

            # 创建输入压力场（中心区域）
            center_x, center_y = self.config.GRID_SIZE // 2, self.config.GRID_SIZE // 2
            half_input = self.config.INPUT_SIZE // 2
            in_pressure = pressure_matrix[center_x - half_input:center_x + half_input, 
                                        center_y - half_input:center_y + half_input]

            # 从文件路径提取雷诺数和时间步
            reynolds_number, time_step = self._extract_metadata(csv_path)
            if reynolds_number is None or time_step is None:
                return None

            data_dict = {
                "memo": {
                    "reynolds": reynolds_number,
                    "time_step": time_step
                },
                "pressure": torch.tensor(pressure_matrix, dtype=torch.float32),
                "in_pressure": torch.tensor(in_pressure, dtype=torch.float32)
            }

            return data_dict, reynolds_number, time_step
            
        except Exception as e:
            logger.error(f"读取文件 {csv_path} 时发生错误: {str(e)}")
            return None
    
    def _extract_metadata(self, csv_path: str) -> Tuple[Optional[str], Optional[float]]:
        """
        从文件路径提取雷诺数和时间步
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            雷诺数和时间步的元组
        """
        try:
            # 从文件夹路径提取雷诺数
            case_folder = os.path.basename(os.path.dirname(csv_path))
            if not case_folder.startswith(self.config.CASE_PREFIX):
                logger.error(f"文件夹名格式不正确: {case_folder}")
                return None, None
                
            reynolds_number = case_folder.split('_')[2]

            # 从文件名提取时间步
            filename = os.path.basename(csv_path)
            if '_t_' not in filename:
                logger.error(f"文件名格式不正确: {filename}")
                return None, None
                
            time_step = float(filename.split('_t_')[1].split('.csv')[0])
            
            return reynolds_number, time_step
            
        except (IndexError, ValueError) as e:
            logger.error(f"提取元数据失败: {str(e)}")
            return None, None


    def normalize_data(self, data_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[float], List[float]]:
        """
        对数据进行归一化处理，按每个case的最小值和最大值进行归一化。
        
        Args:
            data_tensor: 输入数据张量 [time_steps, height, width]
            
        Returns:
            归一化后的数据、最小值列表、最大值列表
        """
        try:
            min_val = data_tensor.min()
            max_val = data_tensor.max()
            
            # 避免除零错误
            if max_val == min_val:
                logger.warning("数据的最大值和最小值相等，归一化结果为零")
                normalized_data = torch.zeros_like(data_tensor)
            else:
                normalized_data = (data_tensor - min_val) / (max_val - min_val)
            
            return normalized_data, min_val.item(), max_val.item()
            
        except Exception as e:
            logger.error(f"归一化过程中发生错误: {str(e)}")
            raise

    def denormalize_data(self, normalized_data: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """
        对数据进行反归一化。
        
        Args:
            normalized_data: 归一化的数据
            min_val: 原始数据的最小值
            max_val: 原始数据的最大值
            
        Returns:
            反归一化后的数据
        """
        try:
            denormalized_data = normalized_data * (max_val - min_val) + min_val
            return denormalized_data
            
        except Exception as e:
            logger.error(f"反归一化过程中发生错误: {str(e)}")
            raise


    def preprocess_and_save_data(self) -> bool:
        """
        遍历根目录下的所有文件夹，处理每个文件夹中的 .csv 文件并合并数据
        
        Returns:
            处理是否成功
        """
        try:
            # 验证根目录存在
            if not os.path.exists(self.config.ROOT_DIRECTORY):
                logger.error(f"根目录不存在: {self.config.ROOT_DIRECTORY}")
                return False

            logger.info(f"开始处理数据，根目录: {self.config.ROOT_DIRECTORY}")
            
            # 测试模式：先验证文件格式
            if self.test_mode:
                logger.info("运行在测试模式，先验证文件格式...")
                if not self._validate_file_formats():
                    logger.error("文件格式验证失败")
                    return False
            
            # 收集所有数据
            self._collect_all_data()
            
            if not self.all_data:
                logger.error("没有找到任何有效数据")
                return False
            
            # 组织和保存数据
            self._organize_and_save_data()
            
            logger.info(f"数据处理完成，共处理 {len(self.all_data)} 个雷诺数的数据")
            return True
            
        except Exception as e:
            logger.error(f"数据处理过程中发生错误: {str(e)}")
            return False
    
    def _collect_all_data(self):
        """
        收集所有CSV文件的数据
        """
        for case_folder in os.listdir(self.config.ROOT_DIRECTORY):
            case_path = os.path.join(self.config.ROOT_DIRECTORY, case_folder)

            if not (os.path.isdir(case_path) and case_folder.startswith(self.config.CASE_PREFIX)):
                continue
                
            reynolds_number = case_folder.split('_')[2]
            logger.info(f"处理雷诺数 {reynolds_number} 的数据")

            # 初始化雷诺数对应的字典
            if reynolds_number not in self.all_data:
                self.all_data[reynolds_number] = {
                    "pressure": [], 
                    "in_pressure": [],
                    "time_steps": []
                }

            # 处理该雷诺数下的所有时间步文件
            self._process_reynolds_data(case_path, reynolds_number)
    
    def _process_reynolds_data(self, case_path: str, reynolds_number: str):
        """
        处理特定雷诺数的所有时间步数据
        
        Args:
            case_path: case文件夹路径
            reynolds_number: 雷诺数
        """
        time_files = []
        
        # 收集所有符合条件的时间步文件
        for time_file in os.listdir(case_path):
            if not (time_file.startswith(self.config.FILE_PREFIX) and time_file.endswith('.csv')):
                continue
                
            try:
                time_step = float(time_file.split('_t_')[1].split('.csv')[0])
                if self.config.START_TIME <= time_step <= self.config.END_TIME:
                    time_files.append((time_file, time_step))
            except (ValueError, IndexError):
                logger.warning(f"无法解析时间步: {time_file}")
                continue
        
        # 按时间步排序
        time_files.sort(key=lambda x: x[1])
        
        # 处理每个时间步文件
        for time_file, time_step in time_files:
            file_path = os.path.join(case_path, time_file)
            result = self.read_csv_data(file_path)
            
            if result is not None:
                data_dict, _, _ = result
                
                # 添加数据到对应的列表
                self.all_data[reynolds_number]["pressure"].append(data_dict['pressure'])
                self.all_data[reynolds_number]["in_pressure"].append(data_dict['in_pressure'])
                self.all_data[reynolds_number]["time_steps"].append(time_step)
                
                logger.debug(f"成功加载: {time_file}, 时间步: {time_step}, 雷诺数: {reynolds_number}")
    
    def _organize_and_save_data(self):
        """
        组织数据并保存到文件
        """
        reynolds_keys = sorted(self.all_data.keys(), key=lambda x: float(x))
        
        # 分别保存每个雷诺数的数据，而不是堆叠所有数据
        reynolds_data_dict = {}
        normalization_params = {
            "pressure": {},
            "in_pressure": {}
        }
        
        valid_reynolds_keys = []
        
        for reynolds_number in reynolds_keys:
            reynolds_data = self.all_data[reynolds_number]
            
            if not reynolds_data["pressure"]:
                logger.warning(f"雷诺数 {reynolds_number} 没有有效数据，跳过")
                continue
            
            try:
                # 堆叠时间序列数据
                pressure_tensor = torch.stack(reynolds_data["pressure"], dim=0)
                in_pressure_tensor = torch.stack(reynolds_data["in_pressure"], dim=0)
                
                # 归一化数据
                norm_pressure, min_p, max_p = self.normalize_data(pressure_tensor)
                norm_in_pressure, min_ip, max_ip = self.normalize_data(in_pressure_tensor)
                
                # 保存归一化参数
                normalization_params["pressure"][reynolds_number] = {"min_val": min_p, "max_val": max_p}
                normalization_params["in_pressure"][reynolds_number] = {"min_val": min_ip, "max_val": max_ip}
                
                # 保存每个雷诺数的数据
                reynolds_data_dict[reynolds_number] = {
                    "pressure": norm_pressure,
                    "in_pressure": norm_in_pressure,
                    "time_steps": reynolds_data["time_steps"],
                    "num_time_steps": len(reynolds_data["time_steps"])
                }
                
                valid_reynolds_keys.append(reynolds_number)
                
                logger.info(f"成功处理雷诺数 {reynolds_number} 的数据，时间步数: {len(reynolds_data['time_steps'])}")
                
            except Exception as e:
                logger.error(f"处理雷诺数 {reynolds_number} 的数据时发生错误: {str(e)}")
                continue
        
        # 检查是否有有效数据
        if not reynolds_data_dict:
            raise ValueError("没有找到任何有效的数据进行处理")
        
        # 保存数据
        save_data = {
            "reynolds_data": reynolds_data_dict,
            "reynolds_keys": valid_reynolds_keys,
            "normalization_params": normalization_params,
            "config": {
                "grid_size": self.config.GRID_SIZE,
                "input_size": self.config.INPUT_SIZE,
                "time_range": (self.config.START_TIME, self.config.END_TIME)
            }
        }
        
        torch.save(save_data, self.config.OUTPUT_FILE)
        logger.info(f"数据已成功保存到 {self.config.OUTPUT_FILE}")
        
        # 打印每个雷诺数的数据形状
        for reynolds_number in valid_reynolds_keys:
            data = reynolds_data_dict[reynolds_number]
            logger.info(f"雷诺数 {reynolds_number}: pressure {data['pressure'].shape}, in_pressure {data['in_pressure'].shape}")
        
        logger.info(f"有效雷诺数: {valid_reynolds_keys}")
    
    def _validate_file_formats(self) -> bool:
        """
        验证所有CSV文件的格式是否正确
        
        Returns:
            验证是否通过
        """
        try:
            logger.info("开始验证文件格式...")
            validation_errors = []
            total_files = 0
            valid_files = 0
            
            for case_folder in os.listdir(self.config.ROOT_DIRECTORY):
                case_path = os.path.join(self.config.ROOT_DIRECTORY, case_folder)
                
                if not (os.path.isdir(case_path) and case_folder.startswith(self.config.CASE_PREFIX)):
                    continue
                
                reynolds_number = case_folder.split('_')[2]
                logger.info(f"验证雷诺数 {reynolds_number} 的文件...")
                
                for time_file in os.listdir(case_path):
                    if not (time_file.startswith(self.config.FILE_PREFIX) and time_file.endswith('.csv')):
                        continue
                    
                    total_files += 1
                    file_path = os.path.join(case_path, time_file)
                    
                    # 验证单个文件
                    if self._validate_single_file(file_path):
                        valid_files += 1
                    else:
                        validation_errors.append(file_path)
            
            logger.info(f"文件验证完成: 总文件数 {total_files}, 有效文件数 {valid_files}")
            
            if validation_errors:
                logger.warning(f"发现 {len(validation_errors)} 个有问题的文件:")
                for error_file in validation_errors[:10]:  # 只显示前10个错误文件
                    logger.warning(f"  - {error_file}")
                if len(validation_errors) > 10:
                    logger.warning(f"  ... 还有 {len(validation_errors) - 10} 个文件有问题")
            
            # 如果有效文件比例太低，返回False
            if total_files > 0 and valid_files / total_files < 0.5:
                logger.error(f"有效文件比例过低: {valid_files}/{total_files} = {valid_files/total_files:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"文件格式验证过程中发生错误: {str(e)}")
            return False
    
    def _validate_single_file(self, file_path: str) -> bool:
        """
        验证单个CSV文件的格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件是否有效
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False
            
            # 读取CSV文件
            data = pd.read_csv(file_path)
            
            # 检查文件是否为空
            if data.empty:
                return False
            
            # 清理列名
            data.columns = data.columns.str.strip()
            
            # 检查必要的列是否存在
            required_columns = ['x', 'y', 'p']
            if not all(col in data.columns for col in required_columns):
                return False
            
            # 检查数据点数量
            expected_size = self.config.GRID_SIZE * self.config.GRID_SIZE
            if len(data) != expected_size:
                return False
            
            # 检查数据类型
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    return False
            
            return True
            
        except Exception:
            return False


class DataVisualizer:
    """
    数据可视化类，用于加载和显示处理后的数据
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
    
    def load_data(self, file_path: str = None) -> Optional[Dict]:
        """
        加载保存的数据文件
        
        Args:
            file_path: 数据文件路径，默认使用配置中的路径
            
        Returns:
            加载的数据字典，如果失败返回None
        """
        try:
            if file_path is None:
                file_path = self.config.OUTPUT_FILE
                
            if not os.path.exists(file_path):
                logger.error(f"数据文件不存在: {file_path}")
                return None
                
            data = torch.load(file_path)
            logger.info(f"成功加载数据文件: {file_path}")
            
            # 为了向后兼容，检查数据格式
            if 'reynolds_data' in data:
                # 新格式
                logger.info("检测到新数据格式")
                logger.info(f"数据包含 {len(data['reynolds_keys'])} 个雷诺数")
                return data
            elif 'pressure' in data and 'in_pressure' in data:
                # 旧格式，转换为新格式
                logger.info("检测到旧数据格式，正在转换...")
                reynolds_data_dict = {}
                reynolds_keys = data['reynolds_keys']
                
                for i, reynolds_number in enumerate(reynolds_keys):
                    reynolds_data_dict[reynolds_number] = {
                        "pressure": data['pressure'][i],
                        "in_pressure": data['in_pressure'][i],
                        "time_steps": [],  # 旧格式可能没有这个信息
                        "num_time_steps": data['pressure'][i].shape[0]
                    }
                
                data['reynolds_data'] = reynolds_data_dict
                logger.info(f"数据包含 {len(data['reynolds_keys'])} 个雷诺数")
                return data
            else:
                logger.error("未知的数据格式")
                return None
                
        except Exception as e:
            logger.error(f"加载数据文件时发生错误: {str(e)}")
            return None
    
    def display_data_overview(self, data: Dict):
        """
        显示数据概览信息
        
        Args:
            data: 加载的数据字典
        """
        try:
            print("\n=== 数据概览 ===")
            print(f"雷诺数: {data['reynolds_keys']}")
            
            # 检查数据格式
            if 'reynolds_data' in data:
                # 新格式
                reynolds_data = data['reynolds_data']
                print("\n各雷诺数数据详情:")
                for reynolds_number in data['reynolds_keys']:
                    if reynolds_number in reynolds_data:
                        rd = reynolds_data[reynolds_number]
                        print(f"  雷诺数 {reynolds_number}:")
                        print(f"    压力数据形状: {rd['pressure'].shape}")
                        print(f"    输入压力数据形状: {rd['in_pressure'].shape}")
                        print(f"    时间步数: {rd['num_time_steps']}")
            else:
                # 旧格式
                print(f"压力数据形状: {data['pressure'].shape}")
                print(f"输入压力数据形状: {data['in_pressure'].shape}")
            
            if 'config' in data:
                config_info = data['config']
                print(f"\n配置信息:")
                print(f"  网格大小: {config_info.get('grid_size', 'N/A')}")
                print(f"  输入大小: {config_info.get('input_size', 'N/A')}")
                print(f"  时间范围: {config_info.get('time_range', 'N/A')}")
            
            # 显示归一化参数信息
            if 'normalization_params' in data:
                norm_params = data['normalization_params']
                print("\n归一化参数:")
                for field in ['pressure', 'in_pressure']:
                    if field in norm_params:
                        print(f"  {field}:")
                        for reynolds_number in data['reynolds_keys']:
                            if reynolds_number in norm_params[field]:
                                params = norm_params[field][reynolds_number]
                                print(f"    雷诺数 {reynolds_number}: min={params['min_val']:.4f}, max={params['max_val']:.4f}")
            
            logger.info("数据概览显示完成")
                
        except Exception as e:
            logger.error(f"显示数据概览时发生错误: {str(e)}")
    
    def display_single_case(self, data: Dict, reynolds_number: str, time_idx: int, save_fig: bool = False):
        """
        显示单个case的数据
        
        Args:
            data: 加载的数据字典
            reynolds_number: 雷诺数
            time_idx: 时间步索引
            save_fig: 是否保存图像
        """
        try:
            reynolds_keys = data['reynolds_keys']
            
            if reynolds_number not in reynolds_keys:
                logger.error(f"雷诺数 {reynolds_number} 不在数据中")
                return
            
            # 检查数据格式
            if 'reynolds_data' in data:
                # 新格式
                reynolds_data = data['reynolds_data'][reynolds_number]
                pressure_data = reynolds_data['pressure']
                in_pressure_data = reynolds_data['in_pressure']
                
                if time_idx >= pressure_data.shape[0]:
                    logger.error(f"时间步索引 {time_idx} 超出范围 (最大: {pressure_data.shape[0]-1})")
                    return
                
                normalized_pressure = pressure_data[time_idx]
                normalized_in_pressure = in_pressure_data[time_idx]
                
                # 获取归一化参数
                norm_params = data['normalization_params']
                min_p = norm_params['pressure'][reynolds_number]['min_val']
                max_p = norm_params['pressure'][reynolds_number]['max_val']
                min_ip = norm_params['in_pressure'][reynolds_number]['min_val']
                max_ip = norm_params['in_pressure'][reynolds_number]['max_val']
                
            else:
                # 旧格式
                reynolds_idx = reynolds_keys.index(reynolds_number)
                normalized_pressure = data['pressure'][reynolds_idx, time_idx]
                normalized_in_pressure = data['in_pressure'][reynolds_idx, time_idx]
                
                # 获取归一化参数
                norm_params = data['normalization_params']
                min_p = norm_params['pressure']['min_vals'][reynolds_idx]
                max_p = norm_params['pressure']['max_vals'][reynolds_idx]
                min_ip = norm_params['in_pressure']['min_vals'][reynolds_idx]
                max_ip = norm_params['in_pressure']['max_vals'][reynolds_idx]
            
            # 反归一化
            denorm_pressure = self.data_processor.denormalize_data(normalized_pressure, min_p, max_p)
            denorm_in_pressure = self.data_processor.denormalize_data(normalized_in_pressure, min_ip, max_ip)
            
            # 创建图像
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 绘制压力场
            im1 = ax1.imshow(denorm_pressure.numpy(), cmap='jet', aspect='equal')
            ax1.set_title(f'Pressure Field (Re={reynolds_number}, Time_idx={time_idx})', fontsize=12)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # 绘制输入压力场
            im2 = ax2.imshow(denorm_in_pressure.numpy(), cmap='jet', aspect='equal')
            ax2.set_title(f'Input Pressure Field (Re={reynolds_number}, Time_idx={time_idx})', fontsize=12)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            
            plt.tight_layout()
            
            if save_fig:
                filename = f'pressure_field_Re{reynolds_number}_t{time_idx}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"图像已保存: {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"显示单个case数据时发生错误: {str(e)}")
    
    def display_all_data_interactive(self, data: Dict, max_display: int = 10):
        """
        交互式显示所有数据（限制显示数量）
        
        Args:
            data: 加载的数据字典
            max_display: 最大显示数量
        """
        try:
            reynolds_keys = data['reynolds_keys']
            
            # 计算总case数
            total_cases = 0
            if 'reynolds_data' in data:
                # 新格式
                for reynolds_number in reynolds_keys:
                    reynolds_data = data['reynolds_data'][reynolds_number]
                    total_cases += reynolds_data['num_time_steps']
            else:
                # 旧格式
                pressure_shape = data['pressure'].shape
                total_cases = pressure_shape[0] * pressure_shape[1]
            
            display_count = min(total_cases, max_display)
            
            logger.info(f"开始显示数据，总共 {total_cases} 个case，将显示前 {display_count} 个")
            
            count = 0
            user_input = ''
            
            for reynolds_number in reynolds_keys:
                if 'reynolds_data' in data:
                    # 新格式
                    reynolds_data = data['reynolds_data'][reynolds_number]
                    num_time_steps = reynolds_data['num_time_steps']
                else:
                    # 旧格式
                    num_time_steps = data['pressure'].shape[1]
                
                for t_idx in range(num_time_steps):
                    if count >= display_count:
                        break
                        
                    logger.info(f"显示第 {count + 1}/{display_count} 个case")
                    self.display_single_case(data, reynolds_number, t_idx)
                    
                    count += 1
                    
                    # 询问用户是否继续
                    if count < display_count:
                        user_input = input("按Enter继续，输入'q'退出: ")
                        if user_input.lower() == 'q':
                            break
                            
                if count >= display_count or (count < display_count and user_input.lower() == 'q'):
                    break
                    
            logger.info("数据显示完成")
            
        except Exception as e:
            logger.error(f"交互式显示数据时发生错误: {str(e)}")


def main():
    """
    主函数 - 提供交互式菜单
    """
    print("="*60)
    print("CFD数据处理和可视化工具")
    print("="*60)
    
    while True:
        print("\n请选择操作:")
        print("1. 处理和保存数据")
        print("2. 创建测试数据并处理")
        print("3. 加载和显示数据")
        print("4. 数据概览")
        print("5. 显示特定case")
        print("6. 验证文件格式")
        print("7. 退出")
        
        choice = input("\n请输入选择 (1-7): ").strip()
        
        if choice == '1':
            config = Config()
            processor = DataProcessor(config, test_mode=True)
            
            print("\n开始处理数据...")
            try:
                success = processor.preprocess_and_save_data()
                if success:
                    print("数据处理完成！")
                else:
                    print("数据处理失败！")
            except Exception as e:
                print(f"处理过程中发生错误: {str(e)}")
                logger.error(f"处理过程中发生错误: {str(e)}")
                
        elif choice == '2':
            print("\n创建测试数据并处理...")
            try:
                quick_process_data(use_test_data=True)
            except Exception as e:
                print(f"测试数据处理过程中发生错误: {str(e)}")
                logger.error(f"测试数据处理过程中发生错误: {str(e)}")
                
        elif choice == '3':
            config = Config()
            visualizer = DataVisualizer(config)
            
            try:
                data = visualizer.load_data()
                if data:
                    visualizer.display_all_data_interactive(data)
                else:
                    print("无法加载数据！")
            except Exception as e:
                print(f"可视化过程中发生错误: {str(e)}")
                logger.error(f"可视化过程中发生错误: {str(e)}")
                
        elif choice == '4':
            config = Config()
            visualizer = DataVisualizer(config)
            
            try:
                data = visualizer.load_data()
                if data:
                    visualizer.display_data_overview(data)
                else:
                    print("无法加载数据！")
            except Exception as e:
                print(f"概览过程中发生错误: {str(e)}")
                logger.error(f"概览过程中发生错误: {str(e)}")
                
        elif choice == '5':
            config = Config()
            visualizer = DataVisualizer(config)
            
            try:
                data = visualizer.load_data()
                if data:
                    reynolds_keys = data['reynolds_keys']
                    print(f"\n可用的雷诺数: {reynolds_keys}")
                    
                    reynolds_number = input("请输入雷诺数: ")
                    time_idx = int(input("请输入时间步索引: "))
                    save_fig = input("是否保存图像? (y/n): ").lower() == 'y'
                    
                    visualizer.display_single_case(data, reynolds_number, time_idx, save_fig)
                else:
                    print("无法加载数据！")
            except Exception as e:
                print(f"显示过程中发生错误: {str(e)}")
                logger.error(f"显示过程中发生错误: {str(e)}")
                
        elif choice == '6':
            config = Config()
            processor = DataProcessor(config, test_mode=True)
            
            print("\n开始验证文件格式...")
            try:
                if processor._validate_file_formats():
                    print("文件格式验证通过！")
                else:
                    print("文件格式验证失败！")
            except Exception as e:
                print(f"验证过程中发生错误: {str(e)}")
                logger.error(f"验证过程中发生错误: {str(e)}")
                
        elif choice == '7':
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入！")


def create_test_data(config: Config):
    """
    创建测试数据用于验证脚本功能
    
    Args:
        config: 配置对象
    """
    import numpy as np
    
    logger.info("创建测试数据...")
    
    # 创建根目录
    os.makedirs(config.ROOT_DIRECTORY, exist_ok=True)
    
    # 创建测试雷诺数和时间步
    test_reynolds = ["100", "200"]
    test_time_steps = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    
    for reynolds in test_reynolds:
        case_folder = f"{config.CASE_PREFIX}{reynolds}"
        case_path = os.path.join(config.ROOT_DIRECTORY, case_folder)
        os.makedirs(case_path, exist_ok=True)
        
        for time_step in test_time_steps:
            # 创建测试数据
            x = np.linspace(0, 1, config.GRID_SIZE)
            y = np.linspace(0, 1, config.GRID_SIZE)
            X, Y = np.meshgrid(x, y)
            
            # 生成模拟压力场（使用正弦波和随机噪声）
            pressure = (np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) + 
                       0.1 * np.random.randn(config.GRID_SIZE, config.GRID_SIZE))
            
            # 展平数据
            x_flat = X.flatten()
            y_flat = Y.flatten()
            p_flat = pressure.flatten()
            
            # 创建DataFrame
            test_data = pd.DataFrame({
                'x': x_flat,
                'y': y_flat,
                'p': p_flat
            })
            
            # 保存CSV文件
            filename = f"{config.FILE_PREFIX}t_{time_step:.2f}.csv"
            file_path = os.path.join(case_path, filename)
            test_data.to_csv(file_path, index=False)
            
        logger.info(f"为雷诺数 {reynolds} 创建了 {len(test_time_steps)} 个测试文件")
    
    logger.info(f"测试数据创建完成，保存在: {config.ROOT_DIRECTORY}")


def quick_process_data(use_test_data: bool = False):
    """
    快速处理数据的函数，用于批处理
    
    Args:
        use_test_data: 是否使用测试数据
    """
    config = Config()
    
    if use_test_data:
        create_test_data(config)
    
    processor = DataProcessor(config, test_mode=True)
    
    logger.info("开始快速数据处理...")
    success = processor.preprocess_and_save_data()
    
    if success:
        logger.info("数据处理成功完成！")
        return True
    else:
        logger.error("数据处理失败！")
        return False


def quick_visualize_data(max_display: int = 3):
    """
    快速可视化数据的函数
    
    Args:
        max_display: 最大显示数量
    """
    config = Config()
    visualizer = DataVisualizer(config)
    
    data = visualizer.load_data()
    if data is not None:
        visualizer.display_data_overview(data)
        visualizer.display_all_data_interactive(data, max_display=max_display)
    else:
        logger.error("无法加载数据文件")


if __name__ == "__main__":
    # 运行主函数
    main()
    
    # 或者直接运行特定功能：
    # quick_process_data(use_test_data=True)  # 快速处理数据（使用测试数据）
    # quick_visualize_data(max_display=3)  # 快速可视化数据
