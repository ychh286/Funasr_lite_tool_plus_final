@echo off
setlocal
pushd "%~dp0" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-FunASRGUI.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo FunASR desktop GUI launch failed. Exit code: %EXIT_CODE%
  echo If the error says PySide6 is missing, run:
  echo   Start-FunASRGUI.cmd -InstallGuiDependency
  pause
)
popd >nul 2>nul
endlocal & exit /b %EXIT_CODE%
