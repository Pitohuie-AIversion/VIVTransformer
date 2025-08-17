# -*- coding: utf-8 -*-
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties

print('[INFO] Python', sys.version)
print('[INFO] Matplotlib', matplotlib.__version__)
print('[INFO] CWD =', os.getcwd())
print('[INFO] MPLCONFIGDIR =', os.environ.get('MPLCONFIGDIR'))
print('[INFO] svg.fonttype =', matplotlib.rcParams.get('svg.fonttype'))
print('[INFO] font.family =', matplotlib.rcParams.get('font.family'))
print('[INFO] font.sans-serif (top 8) =', matplotlib.rcParams.get('font.sans-serif')[:8])

# 1) 列出字体管理器中所有可用字体的家族名
all_names = sorted({f.name for f in fm.fontManager.ttflist})
print('[INFO] Total fonts indexed =', len(all_names))
for target in ['Noto Sans SC','Noto Sans CJK SC','Source Han Sans SC','Microsoft YaHei','SimHei','SimSun','WenQuanYi Micro Hei']:
    print(f"    -> contains '{target}'?", target in all_names)

# 2) 探测项目 fonts 目录内字体的真实家族名
repo_root = os.path.dirname(__file__)
fonts_dir = os.path.join(repo_root, 'fonts')
if os.path.isdir(fonts_dir):
    print('[INFO] fonts dir =', fonts_dir)
    for fname in os.listdir(fonts_dir):
        if fname.lower().endswith(('.ttf','.otf','.ttc')):
            fpath = os.path.join(fonts_dir, fname)
            try:
                fp = FontProperties(fname=fpath)
                print('   - file:', fname, '| family:', fp.get_name())
            except Exception as e:
                print('   - file:', fname, '| family: <ERROR>', e)
else:
    print('[INFO] fonts dir not found')

# 3) findfont 结果
for fam in ['Noto Sans SC','Noto Sans CJK SC','Source Han Sans SC','Microsoft YaHei','SimHei','SimSun']:
    try:
        path = fm.findfont(fam, fallback_to_default=False)
        print(f'[FIND no-fallback] {fam}:', path)
    except Exception as e:
        print(f'[FIND no-fallback] {fam}: ERROR ->', e)
    path_fb = fm.findfont(fam, fallback_to_default=True)
    print(f'[FIND with-fallback] {fam}:', path_fb)

# 4) 实际绘制一次，查看绘制对象所用字体
fig, ax = plt.subplots()
text = ax.text(0.5, 0.5, '中文修复验证：修复-负号-可视化', ha='center', va='center', fontsize=14)
fig.canvas.draw()
try:
    used = text.get_fontname()
except Exception:
    used = 'UNKNOWN'
print('[USED FONTNAME]', used)
plt.close(fig)

print('[DONE]')