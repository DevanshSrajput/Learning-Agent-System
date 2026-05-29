param(
    [int]$Port = 8501,
    [string]$AppFile = "app.py"
)

$ErrorActionPreference = "Stop"

function Ensure-OllamaRunning {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) {
            Write-Host "Ollama already running."
            return
        }
    } catch {
        # Continue to start ollama
    }

    Write-Host "Starting Ollama service..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden | Out-Null

    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) {
                Write-Host "Ollama is ready."
                return
            }
        } catch {}
    }

    Write-Warning "Ollama did not respond in time. Streamlit will still start."
}

Push-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
try {
    Ensure-OllamaRunning
    & ".\.venv\Scripts\python.exe" -m streamlit run $AppFile --server.port $Port
} finally {
    Pop-Location
}
