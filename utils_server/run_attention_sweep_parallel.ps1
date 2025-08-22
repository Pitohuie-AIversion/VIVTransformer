param(
    [string]$ConfigPath = "x:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\generate_data\dynamic_config_server_attention_sweep.yaml",
    [string]$ProjectRoot = "x:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1",
    [int]$Epochs = 1000,
    [int[]]$GPUs = @(0,1),
    [int]$MaxPerGPU = 2
)

# --- Helpers ---
function Get-PythonExe {
    if ($env:CONDA_PYTHON_EXE -and (Test-Path $env:CONDA_PYTHON_EXE)) { return $env:CONDA_PYTHON_EXE }
    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
    throw "未找到 Python 可执行程序，请激活虚拟环境或将 python 加入 PATH"
}

function Get-AttentionLists {
    param(
        [string]$YamlPath
    )
    $py = @"
import json, yaml, sys
from pathlib import Path
p = Path(r"""$YamlPath""")
if not p.exists():
    print(json.dumps({"candidates": [], "skip": []}))
    sys.exit(0)
with open(p, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
model = cfg.get('model', {}) if isinstance(cfg, dict) else {}
cands = model.get('attention_sweep_candidates', []) or []
skip = model.get('attention_skip_list', []) or []
# 统一小写字符串
cands = [str(x).strip().lower() for x in cands if str(x).strip()]
skip  = [str(x).strip().lower() for x in skip if str(x).strip()]
print(json.dumps({"candidates": cands, "skip": skip}, ensure_ascii=False))
"@
    $tmp = Join-Path $env:TEMP "read_yaml_attn_$([Guid]::NewGuid().ToString('N')).py"
    Set-Content -Path $tmp -Value $py -Encoding UTF8
    try {
        $python = Get-PythonExe
        $json = & $python $tmp
        if (-not $LASTEXITCODE -eq 0) { throw "Python 解析 YAML 失败" }
        return ($json | ConvertFrom-Json)
    }
    finally {
        Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue | Out-Null
    }
}

function Start-TrainJob {
    param(
        [int]$GPU,
        [string]$Attention
    )
    $trainer = Join-Path $ProjectRoot "generate_data/dynamic_resolution_trainer.py"
    if (-not (Test-Path $trainer)) { throw "找不到训练脚本：$trainer" }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $attnSafe = $Attention -replace "[^a-z0-9_\-]","_"
    $attnOutRoot = Join-Path $ProjectRoot "results/attention_sweep/$attnSafe"
    New-Item -ItemType Directory -Force -Path (Join-Path $attnOutRoot 'logs') | Out-Null

    # 为子进程设置指定 GPU 的可见性，并运行单注意力的 sweep（内部会写入 results/attention_sweep/<attn>）
    $cmd = @"
`$env:CUDA_VISIBLE_DEVICES='$GPU'; `$env:PYTHONUNBUFFERED='1'; 
& `"$(Get-PythonExe)`" `"$trainer`" --config `"$ConfigPath`" --attention-sweep `"$Attention`" --epochs $Epochs
"@

    $psArgs = @(
        "-NoProfile",
        "-ExecutionPolicy","Bypass",
        "-Command", $cmd
    )

    # 使用当前正在运行的 PowerShell 可执行文件，确保在 Windows/Linux 上均可用（Windows: powershell.exe / Linux: pwsh）
    $psExe = (Get-Process -Id $PID).Path
    $proc = Start-Process -FilePath $psExe -ArgumentList $psArgs -WorkingDirectory $ProjectRoot -PassThru
    return $proc
}

function Plot-AllLossCurves {
    param(
        [string]$OutPath = (Join-Path $ProjectRoot "results/attention_sweep/all_valid_loss_curves.png")
    )
    $py = @"
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(r"""$ProjectRoot""")
attn_dir = root / 'attention_results'
if not attn_dir.exists():
    print(f"未找到目录: {attn_dir}")
    raise SystemExit(0)
attn_types = []
series = {}
for name in sorted(os.listdir(attn_dir)):
    p = attn_dir / name / 'loss_logs' / 'loss_log.txt'
    if not p.is_file():
        continue
    epochs, vals = [], []
    try:
        with open(p, 'r', encoding='utf-8') as f:
            header = True
            for line in f:
                if header:
                    header = False
                    continue
                parts = [x.strip() for x in line.strip().split(',')]
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                epochs.append(int(parts[0]))
                vals.append(float(parts[2]))  # valid loss
        if epochs and vals:
            attn_types.append(name)
            series[name] = (epochs, vals)
    except Exception as e:
        print(f"跳过 {name}: {e}")

if not series:
    print("未收集到任何 loss 曲线，检查 attention_results/*/loss_logs/loss_log.txt 是否存在")
    raise SystemExit(0)

plt.figure(figsize=(16, 9))
for i, (name, (ep, vl)) in enumerate(sorted(series.items())):
    plt.plot(ep, vl, label=name)
plt.xlabel('Epoch')
plt.ylabel('Valid Loss (log10 scale)')
plt.yscale('log', base=10)
plt.title('All Attention Validation Loss Curves')
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
out_path = Path(r"""$OutPath""")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path.as_posix(), dpi=300, bbox_inches='tight', facecolor='white')
print(f"已保存总图：{out_path}")
"@
    $tmp = Join-Path $env:TEMP "plot_all_losses_$([Guid]::NewGuid().ToString('N')).py"
    Set-Content -Path $tmp -Value $py -Encoding UTF8
    try {
        $python = Get-PythonExe
        & $python $tmp
    }
    finally {
        Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue | Out-Null
    }
}

Write-Host "==== 并行注意力训练调度器 ====" -ForegroundColor Cyan
Write-Host "配置文件: $ConfigPath"
Write-Host "项目根目录: $ProjectRoot"
Write-Host "目标 Epochs: $Epochs"
Write-Host "GPU 列表: $($GPUs -join ', ')，每张 GPU 最大并行任务: $MaxPerGPU"

if (-not (Test-Path $ConfigPath)) { throw "配置文件不存在：$ConfigPath" }
if (-not (Test-Path (Join-Path $ProjectRoot 'generate_data/dynamic_resolution_trainer.py'))) { throw "请确认 ProjectRoot 指向 VIVTransformer-1 根目录" }

# 读取注意力候选与跳过列表
$lists = Get-AttentionLists -YamlPath $ConfigPath
$candidates = @()
$skip = @()
if ($lists) {
    $candidates = @($lists.candidates | ForEach-Object { $_.ToLower() })
    $skip = @($lists.skip | ForEach-Object { $_.ToLower() })
}
# 结合内置已知失败集合
$knownFail = @('sk','vip','ufo','muse','aft','a2','shuffle')
$skip = @($skip + $knownFail | Select-Object -Unique)

if (-not $candidates -or $candidates.Count -eq 0) { throw "在 YAML 中未发现 model.attention_sweep_candidates" }

# 过滤得到最终注意力列表
$attentions = @()
foreach ($a in $candidates) { if ($skip -notcontains $a) { $attentions += $a } }
$attentions = $attentions | Select-Object -Unique
if ($attentions.Count -eq 0) { throw "候选经跳过过滤后为空，请检查 YAML" }

Write-Host "发现候选: $($candidates -join ', ')" -ForegroundColor Yellow
Write-Host "跳过列表: $($skip -join ', ')" -ForegroundColor Yellow
Write-Host "实际将运行: $($attentions -join ', ')" -ForegroundColor Green

# 调度结构
$gpuJobs = @{}
$gpuJobInfo = @{}
foreach ($g in $GPUs) { $gpuJobs[$g] = New-Object System.Collections.ArrayList; $gpuJobInfo[$g] = @{} }

# 启动并发任务
$queue = [System.Collections.Queue]::new()
$attentions | ForEach-Object { $queue.Enqueue($_) }

function Get-ActiveCount([int]$g){ ($gpuJobs[$g] | Where-Object { -not $_.HasExited }).Count }

# 主调度循环
while ($queue.Count -gt 0 -or ($GPUs | ForEach-Object { Get-ActiveCount $_ } | Measure-Object -Sum).Sum -gt 0) {
    # 为每张 GPU 补充到最大并行数
    foreach ($g in $GPUs) {
        # 清理已结束的进程引用
        $alive = New-Object System.Collections.ArrayList
        foreach ($p in $gpuJobs[$g]) { if (-not $p.HasExited) { [void]$alive.Add($p) } }
        $gpuJobs[$g] = $alive
        # 启动新任务直到达到上限或队列为空
        while (($gpuJobs[$g].Count -lt $MaxPerGPU) -and ($queue.Count -gt 0)) {
            $attn = $queue.Dequeue()
            Write-Host ("[GPU$g] 启动: {0}" -f $attn)
            $proc = Start-TrainJob -GPU $g -Attention $attn
            [void]$gpuJobs[$g].Add($proc)
            $gpuJobInfo[$g][$proc.Id] = $attn
            Start-Sleep -Milliseconds 200
        }
    }
    Start-Sleep -Seconds 5

    # 打印简单进度
    $counts = ($GPUs | ForEach-Object { Get-ActiveCount $_ })
    Write-Host ("运行中: [" + ($counts -join ", ") + "]，剩余队列: $($queue.Count)")
}

Write-Host "所有任务已完成。开始汇总绘图..." -ForegroundColor Cyan
Plot-AllLossCurves
Write-Host "完成。请查看 results/attention_sweep/all_valid_loss_curves.png 与 attention_results/*/loss_logs/loss_log.txt" -ForegroundColor Green