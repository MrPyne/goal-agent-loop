$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Write-Host "Installed. Run: goal-agent gui -p <target-project>"
