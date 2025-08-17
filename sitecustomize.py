# -*- coding: utf-8 -*-
"""
全局中文字体修复模块 - sitecustomize.py
在Python解释器启动时自动执行，解决Matplotlib中文字符显示问题

功能特性：
- 自动设置中文字体优先级（微软雅黑 > 黑体 > 思源黑体 > DejaVu Sans）
- 修复负号显示问题（axes.unicode_minus = False）
- 过滤字体相关警告信息
- 支持环境变量控制（DISABLE_CHINESE_FONT_FIX=1 可禁用）
"""

import os
import warnings


def setup_chinese_font_global():
    """全局中文字体设置 - 在Python启动时自动执行"""
    
    # 检查是否通过环境变量禁用
    if os.environ.get('DISABLE_CHINESE_FONT_FIX', '').lower() in ('1', 'true', 'yes'):
        return
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        
        # 过滤字体相关警告
        warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
        warnings.filterwarnings('ignore', message='.*does not have a glyph.*')
        warnings.filterwarnings('ignore', message='.*substituting with a dummy symbol.*')
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')
        
        # 设置中文字体优先级列表
        chinese_fonts = [
            'Microsoft YaHei',    # 微软雅黑（Windows常见）
            'SimHei',             # 黑体（Windows系统字体）
            'Source Han Sans SC', # 思源黑体简体中文
            'WenQuanYi Micro Hei', # 文泉驿微米黑（Linux）
            'DejaVu Sans',        # 备用无衬线字体
            'Arial Unicode MS',   # Unicode字体
            'sans-serif'          # 系统默认无衬线字体
        ]
        
        # 应用字体设置
        plt.rcParams['font.family'] = 'sans-serif'
        # 设置中文字体优先列表，确保中文正常显示
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
        
        # 设置其他中文相关参数
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        
    except ImportError:
        # 如果matplotlib未安装，静默跳过
        pass
    except Exception:
        # 其他异常也静默跳过，避免影响程序启动
        pass
    try:
        import matplotlib
        matplotlib.rcParams["svg.fonttype"] = "path"
    except Exception:
        pass


# 自动执行全局字体设置
setup_chinese_font_global()
"""
sitecustomize

全局中文字体修复：当运行本仓库的任何 Python 脚本时，自动为 Matplotlib 启用中文字体支持，
避免中文字符显示为方框/缺失的问题，并修复负号显示问题。

说明：
- 优先使用 Windows 系统常见中文字体：Microsoft YaHei（微软雅黑）
- 其次尝试 SimHei（黑体）、Noto Sans CJK SC、Source Han Sans SC 等
- 如果未索引到字体，尝试从系统字体目录加载 msyh.ttc/simhei.ttf
- 统一设置：font.family = 'sans-serif'，axes.unicode_minus = False

如需禁用此全局修复，可临时设置环境变量：DISABLE_GLOBAL_MPL_CN_FIX=1
"""

import os

# 可选：允许通过环境变量禁用全局修复
if os.environ.get("DISABLE_GLOBAL_MPL_CN_FIX", "0") in {"1", "true", "True"}:
    # 用户显式禁用全局修复
    pass
else:
    try:
        import matplotlib
        from matplotlib import font_manager as fm
        from matplotlib.font_manager import FontProperties

        # 候选中文字体名称（按优先级）
        CANDIDATE_FONTS = [
            "Microsoft YaHei",      # Windows 常见
            "SimHei",               # 黑体
            "SimSun",               # 宋体
            "Noto Sans CJK SC",     # 谷歌开源
            "Source Han Sans SC",   # 思源黑体简体
            "PingFang SC",          # macOS 常见
            "WenQuanYi Micro Hei",  # 文泉驿微米黑（Linux）
            "WenQuanYi Zen Hei",    # 文泉驿
            "Arial Unicode MS",     # 较全的 Unicode 字体（可能未安装）
            "DejaVu Sans",          # 退路（不含全量中文）
        ]

        # 尝试从系统目录注册常见中文字体文件（Windows）
        possible_font_files = [
            r"C:\\Windows\\Fonts\\msyh.ttc",
            r"C:\\Windows\\Fonts\\msyh.ttf",
            r"C:\\Windows\\Fonts\\simhei.ttf",
            r"C:\\Windows\\Fonts\\simsun.ttc",
            r"C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc",
            r"C:\\Windows\\Fonts\\SourceHanSansSC-Regular.otf",
        ]
        # 先注册项目内置字体（如果存在 fonts 目录）
        preferred_from_local_fonts = []
        try:
            repo_root = os.path.dirname(__file__)
            fonts_dir = os.path.join(repo_root, 'fonts')
            if os.path.isdir(fonts_dir):
                for fname in os.listdir(fonts_dir):
                    if fname.lower().endswith((".ttf", ".otf", ".ttc")):
                        fp = os.path.join(fonts_dir, fname)
                        try:
                            fm.fontManager.addfont(fp)
                            try:
                                fam_name = FontProperties(fname=fp).get_name()
                                if fam_name and fam_name not in preferred_from_local_fonts:
                                    preferred_from_local_fonts.append(fam_name)
                            except Exception:
                                pass
                        except Exception:
                            pass
        except Exception:
            pass
        for fp in possible_font_files:
            if os.path.exists(fp):
                try:
                    fm.fontManager.addfont(fp)
                except Exception:
                    pass

        # 构建已可用字体名称集合
        try:
            available_names = {f.name for f in fm.fontManager.ttflist}
        except Exception:
            available_names = set()

        # 选择第一个可用的中文字体
        selected = None
        # 1) 优先选择本地 fonts 目录中识别出的家族名（保持在最前）
        for fam in preferred_from_local_fonts:
            if fam in available_names:
                selected = fam
                break
        # 2) 其次优先这几个标准名字
        if selected is None:
            for pref in ("Noto Sans SC", "Noto Sans CJK SC", "Source Han Sans SC"):
                if pref in available_names:
                    selected = pref
                    break
        # 3) 再按候选名单顺序回退
        if selected is None:
            for name in CANDIDATE_FONTS:
                if name in available_names:
                    selected = name
                    break

        # 如果仍未找到，fallback 到常见可用字体
        if selected is None:
            selected = "Microsoft YaHei"  # 仍设置，若缺失 Matplotlib 会回退，但我们会继续提供候选列表

        # 应用全局 rcParams 设置（尽量在任何 seaborn set_style 之后仍生效）
        # 强制将首选中文字体作为 family，避免回退到 DejaVu Sans
        matplotlib.rcParams["font.family"] = [selected, "sans-serif"]
        # 构造 font.sans-serif 列表：将本地 fonts 目录里识别出的名字置前
        extra_front = [f for f in preferred_from_local_fonts if f != selected]
        matplotlib.rcParams["font.sans-serif"] = [selected] + extra_front + [f for f in CANDIDATE_FONTS if f not in {selected, *extra_front}]
        # 设置负号修复，覆盖matplotlib默认值
        matplotlib.rcParams["axes.unicode_minus"] = False
        # 确保 SVG 输出使用路径化，避免跨平台字体缺失
        matplotlib.rcParams["svg.fonttype"] = "path"

        # 调试输出（按需）
        if os.environ.get("PRINT_MPL_CN_FONT", "0") in {"1", "true", "True"}:
            print(f"[sitecustomize] preferred_from_local_fonts = {preferred_from_local_fonts}")
            print(f"[sitecustomize] available has Noto Sans SC? {'Noto Sans SC' in available_names}")
            print(f"[sitecustomize] available has Source Han Sans SC? {'Source Han Sans SC' in available_names}")
            print(f"[sitecustomize] final selected = {selected}")
            print(f"[sitecustomize] rcParams['font.family'] = {matplotlib.rcParams['font.family']}")
            ss = matplotlib.rcParams.get('font.sans-serif')
            print(f"[sitecustomize] rcParams['font.sans-serif'] (head) = {ss[:8] if isinstance(ss, (list, tuple)) else ss}")
    except Exception:
        # 安静失败，不影响非 Matplotlib 任务
        pass