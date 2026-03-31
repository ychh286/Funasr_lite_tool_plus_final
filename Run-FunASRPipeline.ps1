param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ScriptPath = Join-Path $ProjectRoot 'scripts\run_asr_pipeline.py'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Python virtual environment not found: $PythonExe"
}

Push-Location $ProjectRoot
try {
    & $PythonExe $ScriptPath @CliArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
