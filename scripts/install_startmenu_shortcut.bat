@echo off
setlocal EnableExtensions

chcp 65001 >nul

rem Creates a Start Menu shortcut (idempotent).
rem Folder:  Start Menu\Programs\funasr-lite
rem Name:    funasr-lite WebUI / funasr-lite Desktop GUI

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"

set "TARGET_BAT=%PROJECT_ROOT%\scripts\funasr_lite_web.bat"
set "GUI_TARGET_BAT=%PROJECT_ROOT%\scripts\funasr_lite_gui.bat"

if not exist "%TARGET_BAT%" goto :ERR_NO_TARGET
if not exist "%GUI_TARGET_BAT%" goto :ERR_NO_GUI_TARGET

rem Use PowerShell to resolve the current-user Start Menu\Programs path robustly.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$web='%TARGET_BAT%'; $gui='%GUI_TARGET_BAT%'; $wd='%PROJECT_ROOT%'; $programs=[Environment]::GetFolderPath('Programs'); $folder=Join-Path $programs 'funasr-lite'; New-Item -ItemType Directory -Force -Path $folder | Out-Null; $ws=New-Object -ComObject WScript.Shell; foreach ($item in @(@{Name='funasr-lite WebUI.lnk'; Target=$web}, @{Name='funasr-lite Desktop GUI.lnk'; Target=$gui})) { $lnk=Join-Path $folder $item.Name; if (Test-Path $lnk) { Write-Host '[OK] Shortcut already exists - skipped:' $lnk; continue }; $s=$ws.CreateShortcut($lnk); $s.TargetPath=$item.Target; $s.WorkingDirectory=$wd; $s.WindowStyle=1; $s.Save(); Write-Host '[OK] Created shortcut:' $lnk }"

if errorlevel 1 goto :ERR_PS

echo.
echo You can now open Start Menu and run:
echo   funasr-lite ^> funasr-lite WebUI
echo   funasr-lite ^> funasr-lite Desktop GUI
exit /b 0

:ERR_NO_TARGET
echo [ERROR] Target launcher not found:
echo         "%TARGET_BAT%"
exit /b 1

:ERR_NO_GUI_TARGET
echo [ERROR] GUI launcher not found:
echo         "%GUI_TARGET_BAT%"
exit /b 1

:ERR_PS
echo [ERROR] Failed to create Start Menu shortcut (PowerShell error).
exit /b 1

