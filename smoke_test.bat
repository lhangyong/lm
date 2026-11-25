@echo off
chcp 65001 >nul
title 端到端自检（smoke test）
setlocal
set "PYTHONPATH=%~dp0.deps"
rem 解释器：优先环境变量 PYTHON_EXE；否则读仓库根目录 python_exe.txt（本地自用，已忽略不入库）；最后用 PATH 里的 python
set "PY=%PYTHON_EXE%"
if not defined PY if exist "%~dp0python_exe.txt" for /f "usebackq delims=" %%p in ("%~dp0python_exe.txt") do set "PY=%%p"
if not defined PY set "PY=python"
echo ============================================
echo  端到端自检：模型 / 检测器 / 智能体链路
echo ============================================
"%PY%" "%~dp0scripts\smoke_test.py" %*
endlocal
pause