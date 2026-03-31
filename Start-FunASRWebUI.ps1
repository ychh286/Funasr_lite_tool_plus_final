param(
    [string]$ListenHost = '127.0.0.1',
    [int]$Port = 7860,
    [switch]$Background,
    [switch]$Share
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Python virtual environment not found: $PythonExe"
}

$argList = @('-m', 'funasr_lite.webui', '--host', $ListenHost, '--port', "$Port")
if ($Share) {
    $argList += '--share'
}

if ($Background) {
    $process = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $ProjectRoot -PassThru
    Write-Output "Started FunASR WebUI: PID=$($process.Id) URL=http://$ListenHost`:$Port"
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
