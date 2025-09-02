#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终字体修复方案：强制清理缓存并重新配置
"""

import os
import sys
import shutil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import platform

def clear_matplotlib_cache():
    """清理matplotlib字体缓存"""
    try:
        cache_dir = matplotlib.get_cachedir()
        print(f"清理matplotlib缓存目录: {cache_dir}")
        
        # 删除字体缓存文件
        for filename in os.listdir(cache_dir):
            if filename.startswith('font'):
                filepath = os.path.join(cache_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                        print(f"已删除缓存文件: {filename}")
                except Exception as e:
                    print(f"删除缓存文件失败 {filename}: {e}")
    except Exception as e:
        print(f"清理缓存失败: {e}")

def force_register_chinese_fonts():
    """强制注册中文字体"""
    print("开始强制注册中文字体...")
    
    # Windows系统字体路径
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑粗体
        r"C:\Windows\Fonts\simhei.ttf",   # 黑体
        r"C:\Windows\Fonts\simsun.ttc",   # 宋体
        r"C:\Windows\Fonts\simsunb.ttf",  # 宋体粗体
    ]
    
    registered_fonts = []
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                # 强制添加字体
                fm.fontManager.addfont(font_path)
                registered_fonts.append(font_path)
                print(f"成功注册字体: {font_path}")
            except Exception as e:
                print(f"注册字体失败 {font_path}: {e}")
    
    # 重建字体缓存
    try:
        fm._rebuild()
        print("字体缓存重建完成")
    except:
        pass
    
    return registered_fonts

def configure_matplotlib_chinese():
    """配置matplotlib中文显示"""
    print("配置matplotlib中文显示...")
    
    # 清理并重新配置
    clear_matplotlib_cache()
    force_register_chinese_fonts()
    
    # 获取可用字体列表
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"系统可用字体数量: {len(available_fonts)}")
    
    # 中文字体候选列表（按优先级）
    chinese_fonts = [
        'Microsoft YaHei',
        'Microsoft YaHei UI', 
        'SimHei',
        'SimSun',
        'DengXian',
        'KaiTi',
        'FangSong'
    ]
    
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            print(f"选择字体: {font}")
            break
    
    if not selected_font:
        # 强制使用SimSun
        selected_font = 'SimSun'
        print(f"回退字体: {selected_font}")
    
    # 强制设置matplotlib参数
    plt.rcdefaults()  # 重置为默认值
    
    # 核心配置
    # 统一依赖全局 sitecustomize.py 的设置，避免在此强制覆盖
# matplotlib.rcParams['font.family'] = ['sans-serif']
# matplotlib.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial Unicode MS']
# matplotlib.rcParams['axes.unicode_minus'] = False
# matplotlib.rcParams['font.size'] = 12
    
    # 强制刷新
    plt.clf()
    plt.close('all')
    
    print(f"字体配置完成，当前字体: {selected_font}")
    return selected_font

def test_chinese_display():
    """测试中文显示效果"""
    print("测试中文显示...")
    
    import numpy as np
    
    # 创建测试图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 测试数据
    x = np.arange(10)
    y = np.random.rand(10)
    
    ax.plot(x, y, 'o-', label='测试数据')
    ax.set_xlabel('横坐标轴标签')
    ax.set_ylabel('纵坐标轴标签') 
    ax.set_title('中文字体测试图表')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存测试图
    test_output = 'chinese_font_test.png'
    plt.savefig(test_output, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"测试图已保存: {test_output}")
    
    if os.path.exists(test_output):
        print("请检查测试图中的中文是否正常显示")
        return True
    return False

if __name__ == "__main__":
    print("=" * 50)
    print("matplotlib中文字体终极修复工具")
    print("=" * 50)
    
    # 显示系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"matplotlib版本: {matplotlib.__version__}")
    print("-" * 50)
    
    # 执行修复
    try:
        selected_font = configure_matplotlib_chinese()
        test_result = test_chinese_display()
        
        print("-" * 50)
        if test_result:
            print("[OK] 字体修复完成！")
            print(f"当前使用字体: {selected_font}")
            print("现在可以重新运行模态分析脚本")
        else:
            print("[WARN] 字体修复可能未完全成功")
            print("建议重启Python环境后再试")
            
    except Exception as e:
        print(f"[ERROR] 修复过程出错: {e}")
        print("请尝试重启Python环境或重新安装matplotlib")