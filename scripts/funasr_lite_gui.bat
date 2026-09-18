@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
pushd "%PROJECT_ROOT%" >nul 2>nul
call "%PROJECT_ROOT%\Start-FunASRGUI.cmd" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul 2>nul
endlocal & exit /b %EXIT_CODE%
