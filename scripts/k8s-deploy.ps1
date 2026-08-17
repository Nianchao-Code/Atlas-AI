# Deploy Atlas AI to a local Kubernetes cluster (Docker Desktop / minikube / kind).
param(
  [string]$OpenAIKey = $env:OPENAI_API_KEY,
  [string]$Tag = (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not $OpenAIKey) {
  Write-Host "Set OPENAI_API_KEY or pass -OpenAIKey" -ForegroundColor Yellow
}

function Get-KindNodes {
  $nodes = @(docker ps --filter "label=io.x-k8s.kind.role=control-plane" --format "{{.Names}}")
  if ($nodes) { return $nodes }
  return @(docker ps --format "{{.Names}}" | Where-Object { $_ -match "control-plane|kindest" })
}

function Import-ImageToCluster([string]$Image) {
  $nodes = Get-KindNodes
  if (-not $nodes) {
    Write-Host "No kind node container reachable from this Docker context."
    return $false
  }
  foreach ($node in $nodes) {
    Write-Host "Importing $Image into cluster node $node ..."
    docker save $Image | docker exec -i $node ctr -n k8s.io images import -
  }
  return $true
}

function Show-PodDebug {
  Write-Host "`n--- pod status ---" -ForegroundColor Yellow
  kubectl get pods -n atlas -o wide
  Write-Host "`n--- api events ---" -ForegroundColor Yellow
  kubectl describe pod -l app=api -n atlas | Select-String -Pattern "Image:|State:|Reason:|Failed|Err|Pull" -Context 0,1
}

Write-Host "Building images (tag: $Tag)..."
docker build -t "atlas-api:$Tag" ./backend
docker tag "atlas-api:$Tag" atlas-api:latest
docker build -t "atlas-frontend:$Tag" ./frontend
docker tag "atlas-frontend:$Tag" atlas-frontend:latest

$ctx = kubectl config current-context 2>$null
Write-Host "Kubernetes context: $ctx"

$imported = $false
if ($ctx -match "minikube") {
  minikube image load "atlas-api:$Tag"
  minikube image load "atlas-frontend:$Tag"
  $imported = $true
} elseif ($ctx -match "docker-desktop") {
  # Docker Desktop's cluster reads the same image store the build just wrote to.
  # Its node container is not visible to `docker ps`, so never treat that as a
  # failed import: falling back to :latest here picks up a stale tag that lives
  # only inside the node's containerd and is never refreshed.
  Write-Host "Docker Desktop cluster shares the local image store; no import needed."
  $imported = $true
} else {
  $okApi = Import-ImageToCluster "atlas-api:$Tag"
  $okWeb = Import-ImageToCluster "atlas-frontend:$Tag"
  $imported = [bool]($okApi -and $okWeb)
}

Write-Host "Creating corpus ConfigMap..."
kubectl create namespace atlas --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap atlas-corpus `
  --from-file=./samples/corpus/ `
  -n atlas `
  --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap atlas-eval `
  --from-file=./samples/eval/ `
  -n atlas `
  --dry-run=client -o yaml | kubectl apply -f -

if ($OpenAIKey) {
  kubectl create secret generic atlas-llm `
    --from-literal=OPENAI_API_KEY=$OpenAIKey `
    -n atlas `
    --dry-run=client -o yaml | kubectl apply -f -
}

if ($imported) {
  $apiImage = "atlas-api:${Tag}"
  $webImage = "atlas-frontend:${Tag}"
} else {
  Write-Host "Could not get $Tag into the cluster; falling back to :latest." -ForegroundColor Yellow
  Write-Host "If the cluster does not share this Docker image store, :latest there may be stale." -ForegroundColor Yellow
  docker ps --format "  {{.Names}}  {{.Image}}"
  $apiImage = "atlas-api:latest"
  $webImage = "atlas-frontend:latest"
}

# Render the tag into the manifest instead of applying :latest and correcting it
# with `kubectl set image` afterwards. That left a window where pods were created
# pointing at a tag the cluster may not have, which fails under
# imagePullPolicy: IfNotPresent. The manifest keeps :latest as its standalone
# default so `kubectl apply -f infra/k8s/atlas.yaml` still works on its own.
Write-Host "Applying manifests (api/worker: $apiImage, frontend: $webImage)..."
$manifest = Get-Content ./infra/k8s/atlas.yaml -Raw
$manifest = $manifest.Replace("image: atlas-api:latest", "image: $apiImage")
$manifest = $manifest.Replace("image: atlas-frontend:latest", "image: $webImage")
$rendered = Join-Path ([System.IO.Path]::GetTempPath()) "atlas-rendered.yaml"
[System.IO.File]::WriteAllText($rendered, $manifest, (New-Object System.Text.UTF8Encoding($false)))
kubectl apply -f $rendered

Write-Host "Resetting stuck Redis index stream..."
kubectl rollout status deployment/redis -n atlas --timeout=60s
kubectl exec deploy/redis -n atlas -- redis-cli DEL atlas:index | Out-Null

Write-Host "Forcing Recreate of api/worker/frontend..."
kubectl delete pod -l app=api -n atlas --force --grace-period=0 2>$null | Out-Null
kubectl delete pod -l app=worker -n atlas --force --grace-period=0 2>$null | Out-Null
kubectl delete pod -l app=frontend -n atlas --force --grace-period=0 2>$null | Out-Null

try {
  kubectl rollout status deployment/api -n atlas --timeout=90s
  kubectl rollout status deployment/worker -n atlas --timeout=90s
  kubectl rollout status deployment/frontend -n atlas --timeout=90s
} catch {
  Write-Host "Rollout did not finish in time." -ForegroundColor Yellow
  Show-PodDebug
  throw
}

Write-Host ""
Write-Host "Atlas AI deployed to namespace 'atlas'." -ForegroundColor Green
Write-Host "Frontend: kubectl port-forward svc/frontend 8080:80 -n atlas"
Write-Host "Then open http://127.0.0.1:8080"
Write-Host "Confirm worker: kubectl logs deployment/worker -n atlas --tail=20"
Write-Host "You should see queue=poll-v3. Then Corpus -> Load Kepler sample handbook."
