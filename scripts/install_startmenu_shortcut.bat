@echo off
setlocal EnableExtensions

chcp 65001 >nul

rem Creates a Start Menu shortcut (idempotent).
rem Folder:  Start Menu\Programs\funasr-lite
rem Name:    funasr-lite WebUI

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

set "TARGET_BAT=%PROJECT_ROOT%\scripts\funasr_lite_web.bat"

if not exist "%TARGET_BAT%" goto :ERR_NO_TARGET

rem Use PowerShell to resolve the current-user Start Menu\Programs path robustly.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$target='%TARGET_BAT%'; $wd='%PROJECT_ROOT%'; $programs=[Environment]::GetFolderPath('Programs'); $folder=Join-Path $programs 'funasr-lite'; $lnk=Join-Path $folder 'funasr-lite WebUI.lnk'; if (Test-Path $lnk) { Write-Host '[OK] Shortcut already exists - skipped:' $lnk; exit 0 }; New-Item -ItemType Directory -Force -Path $folder | Out-Null; $ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut($lnk); $s.TargetPath=$target; $s.WorkingDirectory=$wd; $s.WindowStyle=1; $s.Save(); Write-Host '[OK] Created shortcut:' $lnk"

if errorlevel 1 goto :ERR_PS

echo.
echo You can now open Start Menu and run:
echo   funasr-lite ^> funasr-lite WebUI
exit /b 0

:ERR_NO_TARGET
echo [ERROR] Target launcher not found:
echo         "%TARGET_BAT%"
exit /b 1

:ERR_PS
echo [ERROR] Failed to create Start Menu shortcut (PowerShell error).
exit /b 1

