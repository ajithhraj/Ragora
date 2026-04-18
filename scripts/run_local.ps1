$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\\Scripts\\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
  Write-Error "Virtual environment not found at $python"
}

Push-Location $root
try {
  $env:MMRAG_LLM_PROVIDER = "local"
  $env:MMRAG_VECTOR_BACKEND = "faiss"
  $env:MMRAG_AUTH_ENABLED = "false"
  & $python -m uvicorn multimodal_rag.api.app:app --host 127.0.0.1 --port 8000
}
finally {
  Pop-Location
}
