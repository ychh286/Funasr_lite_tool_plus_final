param(
    [switch]$InstallGuiDependency
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$SrcRoot = Join-Path $ProjectRoot 'src'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Python virtual environment not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $SrcRoot)) {
    Write-Error "Project src directory not found: $SrcRoot"
}

if ($InstallGuiDependency) {
    Push-Location $ProjectRoot
    try {
        & $PythonExe -m pip install -e ".[gui]"
        if ($LASTEXITCODE -ne 0) {
            throw "GUI dependency installation failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

$PreviousPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($PreviousPythonPath)) {
    $env:PYTHONPATH = $SrcRoot
}
else {
    $env:PYTHONPATH = "$SrcRoot;$PreviousPythonPath"
}

Push-Location $ProjectRoot
try {
    & $PythonExe -m funasr_lite.gui_entry
    exit $LASTEXITCODE
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
