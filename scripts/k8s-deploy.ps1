# Deploy Atlas AI to a local Kubernetes cluster (Docker Desktop / minikube).
param(
  [string]$OpenAIKey = $env:OPENAI_API_KEY,
  [string]$Tag = (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not $OpenAIKey) {
  Write-Host "Set OPENAI_API_KEY or pass -OpenAIKey" -ForegroundColor Yellow
}

Write-Host "Building images (tag: $Tag)..."
docker build -t "atlas-api:$Tag" ./backend
docker tag "atlas-api:$Tag" atlas-api:latest
docker build -t "atlas-frontend:$Tag" ./frontend
docker tag "atlas-frontend:$Tag" atlas-frontend:latest

$ctx = kubectl config current-context 2>$null
Write-Host "Kubernetes context: $ctx"

if ($ctx -match "minikube") {
  minikube image load atlas-api:latest
  minikube image load atlas-frontend:latest
}

Write-Host "Creating corpus ConfigMap..."
kubectl create namespace atlas --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap atlas-corpus `
  --from-file=./samples/corpus/ `
  -n atlas `
  --dry-run=client -o yaml | kubectl apply -f -

if ($OpenAIKey) {
  kubectl create secret generic atlas-llm `
    --from-literal=OPENAI_API_KEY=$OpenAIKey `
    -n atlas `
    --dry-run=client -o yaml | kubectl apply -f -
}

Write-Host "Applying manifests..."
kubectl apply -f ./infra/k8s/atlas.yaml

Write-Host "Forcing pod recreate with new image..."
kubectl rollout restart deployment/api deployment/worker deployment/frontend -n atlas
kubectl delete pod -l app=worker -n atlas --wait=true 2>$null
kubectl delete pod -l app=api -n atlas --wait=true 2>$null
kubectl rollout status deployment/api -n atlas --timeout=180s
kubectl rollout status deployment/worker -n atlas --timeout=180s
kubectl rollout status deployment/frontend -n atlas --timeout=120s

Write-Host ""
Write-Host "Atlas AI deployed to namespace 'atlas'." -ForegroundColor Green
Write-Host "Frontend: kubectl port-forward svc/frontend 8080:80 -n atlas"
Write-Host "Then open http://127.0.0.1:8080"
Write-Host "API health: kubectl port-forward svc/api 8000:8000 -n atlas  -> http://127.0.0.1:8000/health"
