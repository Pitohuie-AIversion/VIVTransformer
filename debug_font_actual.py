import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib as mpl
import os

# 加载 configure_chinese_font（从 analyze_output_modality_only 复制）
def configure_chinese_font():
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as fm
        import os

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
                    print(f"已注册项目字体: {proj_name} ({pf})")
                    break
                except Exception as e:
                    print(f"项目字体注册失败: {pf}: {e}")
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
                    print(f"已注册系统字体: {fp}")
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

        # 使用 sans-serif 族并由 font.sans-serif 控制优先级，这样缺失字形可回退到 DejaVu Sans 等
        mpl.rcParams["font.family"] = ["sans-serif"]
        if proj_name:
            mpl.rcParams["font.sans-serif"] = [proj_name] + [f for f in candidate_fonts if f != proj_name]
        else:
            mpl.rcParams["font.sans-serif"] = [selected] + [f for f in candidate_fonts if f != selected]
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["svg.fonttype"] = "path"  # 继续保证 SVG 路径化
        mpl.rcParams["text.usetex"] = False

        print(f"已配置中文字体用于 PNG/SVG: {selected}; 备选: {mpl.rcParams['font.sans-serif'][:3]}")
    except Exception as e:
        print(f"字体配置失败: {e}")

print("=== rcParams 调试前 ===")
print(f"font.family: {mpl.rcParams['font.family']}")
print(f"font.sans-serif: {mpl.rcParams['font.sans-serif'][:5]}")

configure_chinese_font()

print("=== rcParams 调试后 ===")
print(f"font.family: {mpl.rcParams['font.family']}")
print(f"font.sans-serif: {mpl.rcParams['font.sans-serif'][:5]}")

# 创建简单图像，包含中文文本
fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, "模态分析", fontsize=20, ha='center', va='center')
ax.text(0.5, 0.3, "输出数据", fontsize=16, ha='center', va='center')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 检查每个文本元素的实际字体
for text in ax.texts:
    font_prop = text.get_fontproperties()
    actual_font = fm.findfont(font_prop)
    print(f"文本 '{text.get_text()}' 实际使用字体文件: {actual_font}")

plt.tight_layout()
plt.savefig("debug_font.png", dpi=150, bbox_inches='tight')
plt.savefig("debug_font.svg", format='svg', bbox_inches='tight')
print("已保存 debug_font.png 和 debug_font.svg")
