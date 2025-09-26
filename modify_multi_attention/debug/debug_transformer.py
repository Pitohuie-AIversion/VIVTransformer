#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
import sys
import os

# 设置文件路径
__file__ = os.path.abspath('run_crop_model_test.py')

try:
    # 执行主脚本
    with open('run_crop_model_test.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # 创建执行环境
    exec_globals = {'__file__': __file__, '__name__': '__main__'}
    exec(code, exec_globals)
    
except Exception as e:
    print("\n" + "="*80)
    print("DETAILED ERROR TRACEBACK:")
    print("="*80)
    traceback.print_exc()
    print("="*80)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("="*80)