@echo off
chcp 65001 >nul
title 训练 PatchCore 无监督模型
setlocal
set "PYTHONPATH=%~dp0.deps"
echo ============================================
echo  螺母缺陷无监督训练 (PatchCore)
echo  仅使用 metal_nut\train\good 正常样本
echo ============================================
"D:\anconda\envs\yolov8-gpu\python.exe" "%~dp0scripts\train.py" %*
endlocal
pause