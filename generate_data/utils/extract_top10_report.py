#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取前10模态分析报告的关键信息
"""

import json
import os
import sys

def extract_top10_info(report_path):
    """从分析报告中提取前10模态的关键信息"""
    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary = {
        'latent_dim': data.get('latent_dim'),
        'input_mse': data.get('reconstruction_error', {}).get('input_mse'),
        'input_rel': data.get('reconstruction_error', {}).get('input_relative_error'),
        'output_mse': data.get('reconstruction_error', {}).get('output_mse'),
        'output_rel': data.get('reconstruction_error', {}).get('output_relative_error'),
        'input_energy_top10': (data.get('input_energy_ratios') or [])[:10],
        'output_energy_top10': (data.get('output_energy_ratios') or [])[:10],
        'input_singular_values_top10': (data.get('input_singular_values') or [])[:10],
        'output_singular_values_top10': (data.get('output_singular_values') or [])[:10],
        'output_dir': os.path.dirname(report_path)
    }
    
    return summary

if __name__ == "__main__":
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        report_path = r"generate_data\utils\hdf5_modality_analysis_top10\modality_analysis_report.json"
    
    try:
        summary = extract_top10_info(report_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)