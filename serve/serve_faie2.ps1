# Start the FAIE2 server on GPU0 only (GPU1 is never visible to it). Logs: serve\logs\faie2_server.*.log
# Usage: .\serve_faie2.ps1 [-Port 8300] [-Ctx 16384] [-Snapshots 4]
param([int] $Port = 8300, [int] $Ctx = 16384, [int] $Snapshots = 4)
$logs = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$pidFile = Join-Path $logs 'faie2_server.pid'
if (Test-Path $pidFile) {
    $old = Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue
    if ($old -and $old.ProcessName -eq 'python') { Stop-Process -Id $old.Id -Force; Start-Sleep 3 }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "port $Port busy" }
$uuid0 = (nvidia-smi --query-gpu=index,uuid --format=csv,noheader | Where-Object { $_ -match '^0,' }).Split(',')[1].Trim()
$env:CUDA_VISIBLE_DEVICES = $uuid0
$py = 'C:\Program Files\Python311\python.exe'
$p = Start-Process -FilePath $py -ArgumentList @((Join-Path $PSScriptRoot 'faie2_server.py'), '--port', "$Port", '--ctx', "$Ctx", '--snapshots', "$Snapshots") `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logs 'faie2_server.out.log') -RedirectStandardError (Join-Path $logs 'faie2_server.err.log')
$p.Id | Set-Content $pidFile
$deadline = (Get-Date).AddSeconds(240)
while ((Get-Date) -lt $deadline) {
    try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).status -eq 'ok') { break } } catch {}
    if ($p.HasExited) { throw "server exited with code $($p.ExitCode); see logs\faie2_server.err.log" }
    Start-Sleep 2
}
$s = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples |
     Where-Object { $_.InstanceName -match "pid_$($p.Id)" } | Measure-Object CookedValue -Sum
"READY pid=$($p.Id) port=$Port shared_mb=$([math]::Round($s.Sum / 1MB, 1))"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
