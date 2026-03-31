@echo off
setlocal enabledelayedexpansion

rem funasr-lite WebUI one-click launcher (Windows)
rem - Uses uv: `uv run funasr-lite-web -- --host <host> --port <port>`
rem - Auto-opens browser to the URL
rem - Close this window (or Ctrl+C) to stop the server

chcp 65001 >nul

rem Resolve project root: <repo>\scripts\this_file.bat -> <repo>
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

rem Defaults (can be overridden by environment variables)
set "HOST=127.0.0.1"
set "PORT=7860"
if not "%FUNASR_LITE_HOST%"=="" set "HOST=%FUNASR_LITE_HOST%"
if not "%FUNASR_LITE_PORT%"=="" set "PORT=%FUNASR_LITE_PORT%"

cd /d "%PROJECT_ROOT%" || (
  echo [ERROR] Failed to cd into project root: "%PROJECT_ROOT%"
  exit /b 1
)

where uv >nul 2>&1
if errorlevel 1 (
  echo [ERROR] `uv` not found in PATH.
  echo         Install uv: https://github.com/astral-sh/uv
  echo         Then run:   uv sync
  pause
  exit /b 1
)

echo [funasr-lite] Project: "%PROJECT_ROOT%"
echo [funasr-lite] Launching WebUI at: http://%HOST%:%PORT%
echo.

rem Open browser with a small delay so the server has time to bind.
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process ('http://%HOST%:%PORT%')" 1>nul 2>nul

rem Start server (blocking). Close window (or Ctrl+C) to stop.
uv run funasr-lite-web --host %HOST% --port %PORT%

endlocal
pause
