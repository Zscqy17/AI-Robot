@echo off
chcp 65001 >nul
echo ============================================
echo   SAM3 + GR-ConvNet 文字驱动抓取系统
echo ============================================
echo.

D:\miniconda3\envs\sam3\python.exe "%~dp0main.py"

pause
