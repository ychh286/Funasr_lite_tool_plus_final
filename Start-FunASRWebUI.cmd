@echo off
setlocal
pushd "%~dp0" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-FunASRWebUI.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo FunASR WebUI launch failed. Exit code: %EXIT_CODE%
  echo Keep this window open and copy the red error text for debugging.
  pause
)
popd >nul 2>nul
endlocal & exit /b %EXIT_CODE%
