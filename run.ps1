param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "creating venv..." -ForegroundColor Cyan
    py -3.11 -m venv (Join-Path $root ".venv")
    & $py -m pip install --upgrade pip
    & $py -m pip install -r (Join-Path $root "requirements.txt")
}

Write-Host "http://localhost:$Port" -ForegroundColor Green
& $py -m uvicorn app.server:app --host 127.0.0.1 --port $Port --app-dir $root
