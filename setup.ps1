param(
    [switch]$PocketTts
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv (Join-Path $ProjectRoot ".venv")
    }
    else {
        & python -m venv (Join-Path $ProjectRoot ".venv")
    }
}

& $VenvPython -m pip install --upgrade pip
$Requirements = if ($PocketTts) {
    Join-Path $ProjectRoot "requirements-pocket-tts.txt"
}
else {
    Join-Path $ProjectRoot "requirements.txt"
}
& $VenvPython -m pip install -r $Requirements

$EnvFile = Join-Path $ProjectRoot ".env"
$ExampleFile = Join-Path $ProjectRoot ".env.example"
if (-not (Test-Path -LiteralPath $EnvFile)) {
    Copy-Item -LiteralPath $ExampleFile -Destination $EnvFile
    Write-Host "Created .env from .env.example. Add your Discord token and model settings."
}

Write-Host "Setup complete. Run .\start.ps1 after configuring .env."
