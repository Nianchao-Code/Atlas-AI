# Paths are relative to this script, not to wherever it was invoked from:
# the previous version resolved ..\.env.example against the caller's directory
# and silently did nothing when that missed.
Set-Location $PSScriptRoot\..
if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
  Write-Host "Created .env from .env.example -- set OPENAI_API_KEY in it." -ForegroundColor Yellow
}

Write-Host "Starting Redis + Qdrant..."
docker compose up -d redis qdrant

Set-Location backend
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

Write-Host "API: http://127.0.0.1:8000  UI: cd frontend && npm run dev"
uvicorn app.main:app --reload --port 8000
