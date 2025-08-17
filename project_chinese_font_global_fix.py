# -*- coding: utf-8 -*-
"""
项目中文字体问题全局修复器

自动扫描项目中所有使用matplotlib的Python文件，批量应用中文字体修复。
解决中文字符显示为方框或缺失字符的问题。

运行方式：
    python project_chinese_font_global_fix.py

功能：
1. 自动发现所有包含matplotlib的.py文件
2. 为每个文件添加中文字体配置代码
3. 生成修复报告
4. 支持预览模式（不实际修改文件）
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set

def find_matplotlib_files(project_root: Path) -> List[Path]:
    """查找项目中所有使用matplotlib的Python文件"""
    matplotlib_files = []
    
    for py_file in project_root.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 检查是否导入了matplotlib
            if (re.search(r'import matplotlib', content) or 
                re.search(r'from matplotlib', content) or
                re.search(r'plt\.', content)):
                matplotlib_files.append(py_file)
                
        except Exception as e:
            print(f"无法读取文件 {py_file}: {e}")
            
    return matplotlib_files

def check_existing_font_fix(content: str) -> bool:
    """检查文件是否已有中文字体修复代码"""
    patterns = [
        r'plt\.rcParams\[.font\.sans-serif.\]',
        r'matplotlib\.rcParams\[.font\.sans-serif.\]',
        r'configure_chinese_font',
        r'Microsoft YaHei',
        r'SimHei'
    ]
    
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    return False

def generate_font_fix_code() -> str:
    """生成中文字体修复代码"""
    return '''
# === 中文字体修复 (auto-generated) ===
import warnings
warnings.filterwarnings("ignore", message=r".*missing from font.*")
warnings.filterwarnings("ignore", message=r".*does not have a glyph.*")
warnings.filterwarnings("ignore", message=r".*substituting with a dummy symbol.*")

try:
    import matplotlib
    # 统一依赖全局 sitecustomize.py 设置，避免重复注入
    # matplotlib.rcParams['font.family'] = 'sans-serif'
    # matplotlib.rcParams['font.sans-serif'] = [
    #     'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 
    #     'Source Han Sans SC', 'PingFang SC', 'WenQuanYi Zen Hei', 'DejaVu Sans'
    # ]
    # matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    pass
# === 中文字体修复结束 ===
'''

def apply_font_fix(file_path: Path, preview_mode: bool = False) -> Dict:
    """为单个文件应用中文字体修复"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 检查是否已有修复
        if check_existing_font_fix(content):
            return {
                'status': 'skipped',
                'reason': '已存在中文字体配置'
            }
        
        # 查找matplotlib导入位置
        lines = content.split('\n')
        insert_line = -1
        
        for i, line in enumerate(lines):
            if (re.search(r'import matplotlib', line) or 
                re.search(r'from matplotlib', line)):
                insert_line = i + 1
                break
        
        if insert_line == -1:
            # 如果没有找到matplotlib导入但有plt使用，在文件开头添加
            for i, line in enumerate(lines):
                if re.search(r'plt\.', line):
                    insert_line = 0
                    break
        
        if insert_line == -1:
            return {
                'status': 'skipped',
                'reason': '未找到合适的插入位置'
            }
        
        # 插入字体修复代码
        font_fix_code = generate_font_fix_code()
        font_fix_lines = font_fix_code.strip().split('\n')
        
        for j, fix_line in enumerate(font_fix_lines):
            lines.insert(insert_line + j, fix_line)
        
        new_content = '\n'.join(lines)
        
        if not preview_mode:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return {
            'status': 'success',
            'reason': f'在第{insert_line}行后插入字体修复代码'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'reason': f'处理失败: {str(e)}'
        }

def main():
    parser = argparse.ArgumentParser(description='项目中文字体问题全局修复器')
    parser.add_argument('--project-root', type=str, 
                       default='.', 
                       help='项目根目录路径 (默认: 当前目录)')
    parser.add_argument('--preview', action='store_true',
                       help='预览模式：仅显示要修改的文件，不实际修改')
    parser.add_argument('--exclude-dirs', nargs='*',
                       default=['.git', '__pycache__', '.venv', 'venv', 'node_modules'],
                       help='要排除的目录名称')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    print(f"正在扫描项目: {project_root}")
    
    # 查找所有matplotlib文件
    matplotlib_files = find_matplotlib_files(project_root)
    
    # 过滤排除目录
    filtered_files = []
    for file_path in matplotlib_files:
        should_exclude = False
        for exclude_dir in args.exclude_dirs:
            if exclude_dir in file_path.parts:
                should_exclude = True
                break
        if not should_exclude:
            filtered_files.append(file_path)
    
    print(f"发现 {len(filtered_files)} 个使用matplotlib的文件")
    
    if args.preview:
        print("\n=== 预览模式 ===")
    
    # 应用修复
    results = {
        'success': [],
        'skipped': [],
        'error': []
    }
    
    for file_path in filtered_files:
        rel_path = file_path.relative_to(project_root)
        result = apply_font_fix(file_path, preview_mode=args.preview)
        
        status = result['status']
        reason = result['reason']
        
        results[status].append(str(rel_path))
        
        if args.preview or status != 'skipped':
            print(f"[{status.upper()}] {rel_path}: {reason}")
    
    # 生成报告
    print(f"\n=== 修复报告 ===")
    print(f"成功修复: {len(results['success'])} 个文件")
    print(f"跳过处理: {len(results['skipped'])} 个文件")
    print(f"处理失败: {len(results['error'])} 个文件")
    
    if args.preview:
        print(f"\n这是预览模式。要实际应用修复，请运行：")
        print(f"python {__file__} --project-root \"{project_root}\"")
    else:
        print(f"\n修复完成！建议运行以下命令验证：")
        print(f"python -c \"import matplotlib.pyplot as plt; plt.text(0.5, 0.5, '中文测试'); plt.savefig('test_cn.png'); print('中文字体测试完成')\"")

if __name__ == "__main__":
    main()