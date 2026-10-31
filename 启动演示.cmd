@echo off
setlocal
chcp 65001 >nul
set "APP_DIR=%~dp0"
set "CAT_PYTHON="
if exist "%APP_DIR%.venv\Scripts\pythonw.exe" set "CAT_PYTHON=%APP_DIR%.venv\Scripts\pythonw.exe"
if not defined CAT_PYTHON if exist "%APP_DIR%..\..\work\cat-env\Scripts\pythonw.exe" set "CAT_PYTHON=%APP_DIR%..\..\work\cat-env\Scripts\pythonw.exe"
if not defined CAT_PYTHON (
  echo 请先按照 README.md 安装 Python 环境。
  pause
  exit /b 1
)
start "" "%CAT_PYTHON%" "%APP_DIR%demo.py"
exit /b 0
