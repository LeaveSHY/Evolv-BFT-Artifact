# ============================================================================
# Evolv-BFT — Local Multi-Node Cluster Runner (Windows PowerShell)
# ============================================================================
# Generates a genesis manifest and starts N local Evolv-BFT nodes.
#
# Usage:
#   .\run_local_cluster.ps1 [-Nodes 4] [-Instances 2] [-BasePort 8080]
#                           [-HttpBase 9000] [-Seed "localtest"]
#                           [-TimeoutMs 1000] [-CheckSecs 10] [-Keep]
# ============================================================================

param(
    [int]$Nodes = 4,
    [int]$Instances = 2,
    [int]$BasePort = 8080,
    [int]$HttpBase = 9000,
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
$GenesisFile = Join-Path $ScriptDir "tmp-genesis-local.json"

$Processes = @()

function Cleanup {
    Write-Host ""
    Write-Host "Stopping all nodes..."
    foreach ($proc in $script:Processes) {
        try {
            if (-not $proc.HasExited) {
                $proc.Kill()
                $proc.WaitForExit(3000)
            }
        } catch {}
    }
    Write-Host "All nodes stopped."
}

# Register cleanup
Register-EngineEvent PowerShell.Exiting -Action { Cleanup } | Out-Null

try {
    # ── Build ───────────────────────────────────────────────────────────────
    Write-Host "=== Building Evolv-BFT ==="
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    Push-Location $SrcDir
    go build -o "$BuildDir\evolvbft.exe" ./cmd/evolvbft
    go build -o "$BuildDir\evolvbft-genesis.exe" ./cmd/evolvbft-genesis
    Pop-Location

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

    # ── Start nodes ─────────────────────────────────────────────────────────
    Write-Host "=== Starting $Nodes nodes ==="
    for ($i = 0; $i -lt $Nodes; $i++) {
        $P2PPort = $BasePort + $i
        $HttpPort = $HttpBase + $i
        $LogFile = Join-Path $ScriptDir "tmp-node-$i.log"

        $proc = Start-Process -FilePath "$BuildDir\evolvbft.exe" `
            -ArgumentList "-id=$i","-port=$P2PPort","-http=$HttpPort",`
                "-manifest=$GenesisFile","-total-nodes=$Nodes",`
                "-initial-validators=$Nodes","-instances=$Instances",`
                "-timeout-ms=$TimeoutMs" `
            -RedirectStandardOutput $LogFile `
            -RedirectStandardError (Join-Path $ScriptDir "tmp-node-$i.err.log") `
            -PassThru -NoNewWindow

        $Processes += $proc
        Write-Host "  Node $i : PID=$($proc.Id), P2P=:$P2PPort, HTTP=:$HttpPort, log=$LogFile"
    }

    Write-Host ""
    Write-Host "=== Waiting ${CheckSecs}s for consensus to start ==="
    Start-Sleep -Seconds $CheckSecs

    # ── Check health ────────────────────────────────────────────────────────
    Write-Host "=== Checking node health ==="
    $AllOk = $true
    for ($i = 0; $i -lt $Nodes; $i++) {
        $HttpPort = $HttpBase + $i
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$HttpPort/metrics" -TimeoutSec 3 -ErrorAction Stop
            $committed = 0
            if ($resp -is [PSCustomObject] -and $resp.PSObject.Properties.Name -contains "committed_blocks") {
                $committed = $resp.committed_blocks
            } elseif ($resp -is [string] -and $resp -match '"committed_blocks":(\d+)') {
                $committed = [int]$Matches[1]
            }
            Write-Host "  Node $i : committed_blocks=$committed"
            if ($committed -eq 0) { $AllOk = $false }
        } catch {
            Write-Host "  Node $i : UNREACHABLE ($($_.Exception.Message))"
            $AllOk = $false
        }
    }

    Write-Host ""
    if ($AllOk) {
        Write-Host "OK All nodes have committed blocks - consensus is working!" -ForegroundColor Green
    } else {
        Write-Host "WARNING Some nodes have not committed blocks yet." -ForegroundColor Yellow
        Write-Host "   Check logs in $ScriptDir\tmp-node-*.log"
    }

    if ($Keep) {
        Write-Host ""
        Write-Host "=== Running (Ctrl+C to stop) ==="
        Write-Host "Press Ctrl+C to stop all nodes..."
        try {
            while ($true) { Start-Sleep -Seconds 1 }
        } catch {
            # Ctrl+C pressed
        }
    }
} finally {
    Cleanup
}
