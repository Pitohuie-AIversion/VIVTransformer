@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ================================================================
REM 配置管理工具启动脚本 (Windows)
REM 
REM 功能: 快速启动配置管理工具
REM 支持: 参数复用、配置生成、批量实验
REM 作者: AI Assistant
REM 日期: 2025
REM ================================================================

echo.
echo ================================================================
echo 🚀 VIVTransformer 配置管理工具启动器
echo ================================================================
echo 功能: 配置文件参数复用和管理
echo 支持: 快速修改、模板生成、批量实验
echo ================================================================
echo.

REM 检查Python环境
echo 🔍 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python环境
    echo 请确保已安装Python并添加到PATH环境变量
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ 找到Python: !PYTHON_VERSION!

REM 检查必需的Python包
echo.
echo 🔍 检查必需的Python包...

set REQUIRED_PACKAGES=yaml pathlib
set MISSING_PACKAGES=

for %%p in (%REQUIRED_PACKAGES%) do (
    python -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        set MISSING_PACKAGES=!MISSING_PACKAGES! %%p
    ) else (
        echo ✅ %%p 已安装
    )
)

if not "!MISSING_PACKAGES!"=="" (
    echo.
    echo ❌ 缺少必需的Python包:!MISSING_PACKAGES!
    echo.
    echo 🔧 正在安装缺少的包...
    for %%p in (!MISSING_PACKAGES!) do (
        echo 安装 %%p...
        pip install %%p
        if errorlevel 1 (
            echo ❌ 安装 %%p 失败
            echo 请手动运行: pip install %%p
            pause
            exit /b 1
        )
    )
    echo ✅ 所有包安装完成
)

REM 检查配置文件
echo.
echo 📁 检查配置文件...

set CONFIG_FILES=dynamic_config_server_downsampling_optimized.yaml dynamic_config_server_downsampling.yaml
set FOUND_CONFIG=0

for %%f in (%CONFIG_FILES%) do (
    if exist "%%f" (
        echo ✅ 找到配置文件: %%f
        set FOUND_CONFIG=1
    )
)

if !FOUND_CONFIG!==0 (
    echo ⚠️  警告: 未找到默认配置文件
    echo 你仍然可以使用工具，但需要手动指定配置文件路径
)

REM 检查工具文件
echo.
echo 🔧 检查工具文件...

set TOOL_FILES=config_manager.py quick_config.py
set MISSING_TOOLS=

for %%f in (%TOOL_FILES%) do (
    if exist "%%f" (
        echo ✅ %%f 存在
    ) else (
        echo ❌ %%f 不存在
        set MISSING_TOOLS=!MISSING_TOOLS! %%f
    )
)

if not "!MISSING_TOOLS!"=="" (
    echo.
    echo ❌ 错误: 缺少必需的工具文件:!MISSING_TOOLS!
    echo 请确保所有文件都在当前目录中
    echo.
    pause
    exit /b 1
)

REM 显示启动选项
echo.
echo ================================================================
echo 🚀 选择启动模式
echo ================================================================
echo 1. 🎯 快速配置工具 (交互式界面)
echo 2. 🔧 配置管理器 (命令行工具)
echo 3. 📖 运行使用示例
echo 4. ✅ 验证配置文件
echo 5. 📊 查看配置摘要
echo 6. ❓ 显示帮助信息
echo 0. 🚪 退出
echo ================================================================
echo.

set /p CHOICE="请选择启动模式 (0-6): "

if "%CHOICE%"=="0" (
    echo 👋 退出程序
    goto :end
)

if "%CHOICE%"=="1" (
    echo.
    echo 🎯 启动快速配置工具...
    echo ================================================================
    python quick_config.py
    goto :end
)

if "%CHOICE%"=="2" (
    echo.
    echo 🔧 配置管理器命令行工具
    echo ================================================================
    echo 可用操作: validate, summary, compare, override, template, experiment
    echo.
    echo 示例命令:
    echo   python config_manager.py -c config.yaml -a summary
    echo   python config_manager.py -c config.yaml -a validate
    echo.
    echo 更多帮助: python config_manager.py --help
    echo ================================================================
    echo.
    cmd /k "echo 配置管理器命令行环境已准备就绪"
    goto :end
)

if "%CHOICE%"=="3" (
    echo.
    echo 📖 运行使用示例...
    echo ================================================================
    if exist "config_usage_examples.py" (
        python config_usage_examples.py
    ) else (
        echo ❌ 找不到 config_usage_examples.py 文件
    )
    goto :end
)

if "%CHOICE%"=="4" (
    echo.
    echo ✅ 验证配置文件
    echo ================================================================
    
    REM 查找配置文件
    set VALIDATE_CONFIG=
    for %%f in (%CONFIG_FILES%) do (
        if exist "%%f" (
            set VALIDATE_CONFIG=%%f
            goto :validate_found
        )
    )
    
    :validate_found
    if "!VALIDATE_CONFIG!"=="" (
        echo ❌ 未找到配置文件
        set /p VALIDATE_CONFIG="请输入配置文件路径: "
    ) else (
        echo 🔍 验证配置文件: !VALIDATE_CONFIG!
    )
    
    if exist "!VALIDATE_CONFIG!" (
        python config_manager.py -c "!VALIDATE_CONFIG!" -a validate
    ) else (
        echo ❌ 配置文件不存在: !VALIDATE_CONFIG!
    )
    goto :end
)

if "%CHOICE%"=="5" (
    echo.
    echo 📊 查看配置摘要
    echo ================================================================
    
    REM 查找配置文件
    set SUMMARY_CONFIG=
    for %%f in (%CONFIG_FILES%) do (
        if exist "%%f" (
            set SUMMARY_CONFIG=%%f
            goto :summary_found
        )
    )
    
    :summary_found
    if "!SUMMARY_CONFIG!"=="" (
        echo ❌ 未找到配置文件
        set /p SUMMARY_CONFIG="请输入配置文件路径: "
    ) else (
        echo 📊 分析配置文件: !SUMMARY_CONFIG!
    )
    
    if exist "!SUMMARY_CONFIG!" (
        python config_manager.py -c "!SUMMARY_CONFIG!" -a summary
    ) else (
        echo ❌ 配置文件不存在: !SUMMARY_CONFIG!
    )
    goto :end
)

if "%CHOICE%"=="6" (
    echo.
    echo ❓ 配置管理工具帮助信息
    echo ================================================================
    echo.
    echo 🚀 VIVTransformer 配置管理工具
    echo.
    echo 📋 主要功能:
    echo   • 配置文件参数复用和模板管理
    echo   • 快速修改常用参数（批次大小、学习率等）
    echo   • 硬件优化配置生成
    echo   • 批量实验配置生成
    echo   • 配置文件验证和比较
    echo   • 分辨率预设应用
    echo.
    echo 🎯 使用建议:
    echo   1. 首先运行"快速配置工具"了解基本功能
    echo   2. 使用"验证配置文件"检查配置正确性
    echo   3. 使用"查看配置摘要"了解配置详情
    echo   4. 根据需要生成优化或实验配置
    echo.
    echo 📁 生成的文件:
    echo   • config_*.yaml - 各种配置文件
    echo   • experiment_*/ - 实验配置目录
    echo   • config_comparison_result.json - 配置比较结果
    echo.
    echo 💡 提示:
    echo   • 支持YAML锚点和引用的参数复用
    echo   • 自动验证生成的配置文件
    echo   • 支持嵌套参数路径修改
    echo   • 提供多种硬件环境优化模板
    echo.
    echo ================================================================
    goto :end
)

echo ❌ 无效选择: %CHOICE%
echo 请输入 0-6 之间的数字

:end
echo.
echo ================================================================
echo 感谢使用 VIVTransformer 配置管理工具！
echo ================================================================
echo.
pause