@echo off
chcp 65001 >nul
title 启动网页版检测系统（螺母 PatchCore）
setlocal
set "PYTHONPATH=%~dp0.deps"
echo ============================================
echo  启动中... 浏览器访问 http://127.0.0.1:7860
echo  稍后会自动打开浏览器，关闭本窗口即停止服务
echo ============================================
"D:\anconda\envs\yolov8-gpu\python.exe" "%~dp0agent\web.py" %*
endlocal
pause