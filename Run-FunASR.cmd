@echo off
setlocal
pushd "%~dp0" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-FunASR.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul 2>nul
endlocal & exit /b %EXIT_CODE%
