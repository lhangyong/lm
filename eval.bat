@echo off
chcp 65001 >nul
title 评估模型
setlocal
set "PYTHONPATH=%~dp0.deps"
echo ============================================
echo  测试集评估（图像级 + 像素级 AUROC）
echo ============================================
"D:\anconda\envs\yolov8-gpu\python.exe" "%~dp0scripts\evaluate.py" %*
endlocal
pause