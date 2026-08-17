# Copy to .env in repo root
Copy-Item ..\.env.example ..\.env -ErrorAction SilentlyContinue
Set-Location $PSScriptRoot\..

Write-Host "Starting Redis + Qdrant..."
docker compose up -d redis qdrant

Set-Location backend
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

Write-Host "API: http://127.0.0.1:8000  UI: cd frontend && npm run dev"
uvicorn app.main:app --reload --port 8000
