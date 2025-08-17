# -*- coding: utf-8 -*-
"""
子目录自动中文字体修复 - sitecustomize.py (utils)
确保从本目录直接运行的脚本也能自动启用 Matplotlib 中文字体支持。
"""
import os
import warnings

def _setup_cn_font():
    # 允许通过环境变量禁用
    if os.environ.get('DISABLE_CHINESE_FONT_FIX', '').lower() in ('1','true','yes'):
        return
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        # 降噪：忽略字体警告
        warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
        warnings.filterwarnings('ignore', message='.*does not have a glyph.*')
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
        # 尝试注册常见中文字体（Windows 优先）
        for fp in (
            r"C:\\Windows\\Fonts\\msyh.ttc",
            r"C:\\Windows\\Fonts\\msyh.ttf",
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:\\Windows\\Fonts\\simsun.ttc",
        ):
            if os.path.exists(fp):
                try:
                    fm.fontManager.addfont(fp)
                except Exception:
                    pass
        # 构建候选列表并选择可用字体
        candidates = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'Source Han Sans SC',
            'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS'
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        selected = next((f for f in candidates if f in available), 'Microsoft YaHei')
        # 全局 rcParams
        matplotlib.rcParams['font.family'] = 'sans-serif'
        # 启用中文字体优先列表与负号、SVG路径化设置，避免中文显示为方框
        matplotlib.rcParams['font.sans-serif'] = [selected] + [f for f in candidates if f != selected]
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['svg.fonttype'] = 'path'
        if os.environ.get('PRINT_MPL_CN_FONT','0') in {'1','true','True'}:
            print(f"[utils.sitecustomize] Matplotlib 中文字体: {selected}")
    except Exception:
        pass

_setup_cn_font()