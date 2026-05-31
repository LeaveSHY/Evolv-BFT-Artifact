# ============================================================================
# Evolv-BFT — Integrated Go + Python MARL Cluster Runner (Windows PowerShell)
# ============================================================================
# Launches the Python SFAC service, then starts N Go consensus nodes connected
# to it via the narrow HTTP interface (d=5 features per agent per epoch).
#
# Usage:
#   .\run_integrated.ps1 [-Nodes 4] [-Instances 2] [-BasePort 8080]
#                        [-HttpBase 9000] [-MarlPort 18080] [-Seed "localtest"]
#                        [-TimeoutMs 1000] [-CheckSecs 10] [-Keep]
# ============================================================================

param(
    [int]$Nodes = 4,
    [int]$Instances = 2,
    [int]$BasePort = 8080,
    [int]$HttpBase = 9000,
    [int]$MarlPort = 18080,
    [string]$Seed = "localtest",
    [int]$TimeoutMs = 1000,
    [int]$CheckSecs = 10,
    [switch]$Keep
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$SrcDir = Join-Path $RootDir "src"
$BuildDir = Join-Path $RootDir "build"
$MarlDir = Join-Path $RootDir "marl"
$GenesisFile = Join-Path $ScriptDir "tmp-genesis-local.json"

$Processes = @()
$MarlProcess = $null

function Cleanup {
    Write-Host ""
    Write-Host "Stopping all processes..."
    foreach ($proc in $script:Processes) {
        try {
            if (-not $proc.HasExited) {
                $proc.Kill()
                $proc.WaitForExit(3000)
            }
        }
        catch {}
    }
    if ($script:MarlProcess -and -not $script:MarlProcess.HasExited) {
        try {
            $script:MarlProcess.Kill()
            $script:MarlProcess.WaitForExit(3000)
        }
        catch {}
    }
    Write-Host "All processes stopped."
}

Register-EngineEvent PowerShell.Exiting -Action { Cleanup } | Out-Null

try {
    # ── Build Go binaries ───────────────────────────────────────────────────
    Write-Host "=== Building Evolv-BFT Go binaries ==="
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    Push-Location $SrcDir
    go build -o "$BuildDir\evolvbft.exe" ./cmd/evolvbft
    go build -o "$BuildDir\evolvbft-genesis.exe" ./cmd/evolvbft-genesis
    Pop-Location

    # ── Check Python environment ────────────────────────────────────────────
    Write-Host "=== Checking Python MARL environment ==="
    $VenvDir = Join-Path $RootDir ".venv_marl"
    if (Test-Path (Join-Path $VenvDir "Scripts\python.exe")) {
        $PythonExe = Join-Path $VenvDir "Scripts\python.exe"
        Write-Host "Using existing venv at $VenvDir"
    }
    else {
        $PythonExe = "python"
        Write-Host "No venv found, using system Python. Run: python -m venv .venv_marl && .venv_marl\Scripts\pip install -r marl\requirements.txt"
    }

    # ── Start Python MARL service ───────────────────────────────────────────
    Write-Host "=== Starting SFAC MARL service on port $MarlPort ==="
    $MarlLogFile = Join-Path $ScriptDir "tmp-marl-service.log"
    $script:MarlProcess = Start-Process -FilePath $PythonExe `
        -ArgumentList "-m", "uvicorn", "marl.app:app", "--host", "0.0.0.0", "--port", "$MarlPort", "--log-level", "info" `
        -WorkingDirectory $RootDir `
        -RedirectStandardOutput $MarlLogFile `
        -RedirectStandardError (Join-Path $ScriptDir "tmp-marl-service-err.log") `
        -PassThru -NoNewWindow

    # Wait for MARL service to be ready
    Write-Host "Waiting for MARL service to become ready..."
    $MarlUrl = "http://127.0.0.1:$MarlPort/health"
    $MaxRetries = 30
    $Retries = 0
    while ($Retries -lt $MaxRetries) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-RestMethod -Uri $MarlUrl -Method Get -TimeoutSec 2
            if ($resp.ok) {
                Write-Host "MARL service ready (model_ready=$($resp.model_ready))"
                break
            }
        }
        catch {}
        $Retries++
    }
    if ($Retries -ge $MaxRetries) {
        Write-Host "WARNING: MARL service did not respond after $MaxRetries seconds. Continuing anyway..."
    }

    # ── Generate genesis manifest ───────────────────────────────────────────
    Write-Host "=== Generating genesis manifest (nodes=$Nodes, seed=$Seed) ==="
    & "$BuildDir\evolvbft-genesis.exe" `
        -nodes=$Nodes `
        -seed="$Seed" `
        -base-host="127.0.0.1" `
        -base-port=$BasePort `
        -verbose `
        -out="$GenesisFile"

    Write-Host ""

    # ── Start Go consensus nodes ────────────────────────────────────────────
    $MarlInferUrl = "http://127.0.0.1:$MarlPort/infer"
    Write-Host "=== Starting $Nodes consensus nodes (adaptive-policy=facmac-http) ==="
    for ($i = 0; $i -lt $Nodes; $i++) {
        $P2PPort = $BasePort + $i
        $HttpPort = $HttpBase + $i
        $LogFile = Join-Path $ScriptDir "tmp-node-$i.log"

        $proc = Start-Process -FilePath "$BuildDir\evolvbft.exe" `
            -ArgumentList "-id=$i", "-port=$P2PPort", "-http=$HttpPort", `
            "-manifest=$GenesisFile", "-total-nodes=$Nodes", `
            "-initial-validators=$Nodes", "-instances=$Instances", `
            "-timeout-ms=$TimeoutMs", `
            "-adaptive-enabled", `
            "-adaptive-policy=facmac-http", `
            "-adaptive-policy-url=$MarlInferUrl", `
            "-adaptive-interval-ms=5000", `
            "-adaptive-trace-path=traces/node-$i-trace.jsonl", `
            "-consensus-topic=evolvbft-consensus" `
            -RedirectStandardOutput $LogFile `
            -RedirectStandardError (Join-Path $ScriptDir "tmp-node-$i-err.log") `
            -PassThru -NoNewWindow
        $script:Processes += $proc
        Write-Host "  Node $i: PID=$($proc.Id) P2P=:$P2PPort HTTP=:$HttpPort -> MARL=$MarlInferUrl"
    }

    # ── Health monitoring ───────────────────────────────────────────────────
    Write-Host ""
    Write-Host "=== Integrated cluster running ==="
    Write-Host "  MARL service: http://127.0.0.1:$MarlPort  (PID=$($script:MarlProcess.Id))"
    Write-Host "  Consensus nodes: $Nodes nodes on ports $HttpBase-$($HttpBase + $Nodes - 1)"
    Write-Host "  Press Ctrl+C to stop all."
    Write-Host ""

    if ($CheckSecs -gt 0) {
        Write-Host "Checking health every $CheckSecs seconds..."
        while ($true) {
            Start-Sleep -Seconds $CheckSecs

            # Check MARL service
            $marlAlive = $false
            try {
                $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$MarlPort/health" -Method Get -TimeoutSec 2
                $marlAlive = $resp.ok
            }
            catch {}

            # Check consensus nodes
            $nodesAlive = 0
            for ($i = 0; $i -lt $Nodes; $i++) {
                try {
                    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$($HttpBase + $i)/metrics" -Method Get -TimeoutSec 2
                    $nodesAlive++
                }
                catch {}
            }

            $ts = Get-Date -Format "HH:mm:ss"
            Write-Host "[$ts] MARL=$marlAlive  Nodes=$nodesAlive/$Nodes alive"

            # Check if any critical process died
            if ($script:MarlProcess.HasExited) {
                Write-Host "MARL service exited (code=$($script:MarlProcess.ExitCode)). Stopping cluster."
                break
            }
        }
    }
    else {
        while ($true) { Start-Sleep -Seconds 60 }
    }

}
catch {
    Write-Host "Error: $_"
}
finally {
    if (-not $Keep) {
        Cleanup
    }
    else {
        Write-Host "Keeping processes alive (-Keep flag). Remember to stop them manually."
    }
}
