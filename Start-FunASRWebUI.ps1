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

$ModuleName = if ($Legacy) { 'funasr_lite.webui' } else { 'funasr_lite.webui_workbench' }
$argList = @('-m', $ModuleName, '--host', $ListenHost, '--port', "$Port")
if ($Share) {
    $argList += '--share'
}

if ($Background) {
    $process = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $ProjectRoot -PassThru
    $ModeName = if ($Legacy) { 'Legacy WebUI' } else { 'Audio Workbench V2' }
    Write-Output "Started FunASR $ModeName: PID=$($process.Id) URL=http://$ListenHost`:$Port"
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
