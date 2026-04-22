param(
    [string]$ListenHost = '127.0.0.1',
    [int]$Port = 7860,
    [switch]$Background,
    [switch]$Share,
    [switch]$Legacy
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Python virtual environment not found: $PythonExe"
}

$ModuleName = 'funasr_lite.webui_workbench'
if ($Legacy) {
    $ModuleName = 'funasr_lite.webui'
}

$argList = @('-m', $ModuleName, '--host', $ListenHost, '--port', "$Port")
if ($Share) {
    $argList += '--share'
}

if ($Background) {
    $process = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $ProjectRoot -PassThru
    Write-Output "Started FunASR WebUI: PID=$($process.Id) URL=http://$ListenHost`:$Port Module=$ModuleName"
    exit 0
}

Push-Location $ProjectRoot
try {
    & $PythonExe @argList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
