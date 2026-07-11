$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }

Push-Location $ProjectRoot
try {
    & $Python (Join-Path $ProjectRoot "bot.py")
}
finally {
    Pop-Location
}
