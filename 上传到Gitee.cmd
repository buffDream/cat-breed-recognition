@echo off
setlocal
chcp 65001 >nul
set "APP_DIR=%~dp0"
set "CAT_PYTHON="
if exist "%APP_DIR%.venv\Scripts\python.exe" set "CAT_PYTHON=%APP_DIR%.venv\Scripts\python.exe"
if not defined CAT_PYTHON if exist "%APP_DIR%..\..\work\cat-env\Scripts\python.exe" set "CAT_PYTHON=%APP_DIR%..\..\work\cat-env\Scripts\python.exe"
if defined CAT_PYTHON (
  "%CAT_PYTHON%" -X utf8 "%APP_DIR%upload_gitee.py"
) else (
  python -X utf8 "%APP_DIR%upload_gitee.py"
)
exit /b 0
