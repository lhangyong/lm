@echo off
chcp 65001 >nul
title 端到端自检（smoke test）
setlocal
echo ============================================
echo  端到端自检：模型 / 检测器 / 智能体链路
echo ============================================
"D:\anconda\envs\yolov8-gpu\python.exe" "%~dp0scripts\smoke_test.py" %*
endlocal
pause
