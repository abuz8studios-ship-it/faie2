# Launch a llama-server benchmark target on GPU0 only and report memory spill.
# Usage: .\serve.ps1 -Model <gguf> -Tag <log name> [-Draft <gguf>] [-SpecType draft-dflash] [-NMax 7] [-Extra @(...)]
# Stops any server this script started before (bench\logs\server.pid). GPU1 is never visible to the process.
param(
    [Parameter(Mandatory)] [string] $Model,
    [Parameter(Mandatory)] [string] $Tag,
    [string] $Draft,
    [string] $SpecType,
    [int] $NMax = 7,
    [string] $Exe = 'E:\ABU\MODELS\runtime\src\llama.cpp-dflash2\build\bin\llama-server.exe',
    [string[]] $Extra = @()
)
$logs = Join-Path $PSScriptRoot 'logs'
$pidFile = Join-Path $logs 'server.pid'
if (Test-Path $pidFile) { Stop-Process -Id (Get-Content $pidFile) -Force -ErrorAction SilentlyContinue; Start-Sleep 3 }
if (Get-NetTCPConnection -LocalPort 8191 -State Listen -ErrorAction SilentlyContinue) { throw 'port 8191 busy' }

$uuid0 = (nvidia-smi --query-gpu=index,uuid --format=csv,noheader | Where-Object { $_ -match '^0,' }).Split(',')[1].Trim()
$env:CUDA_VISIBLE_DEVICES = $uuid0

$a = @('-m', $Model, '--host', '127.0.0.1', '--port', '8191', '--device', 'CUDA0', '-ngl', '99', '-fa', 'on',
       '-c', '4096', '-b', '512', '-ub', '256', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
       '--parallel', '1', '--jinja', '--no-warmup', '--metrics')
if ($Draft) { $a += @('-md', $Draft, '--device-draft', 'CUDA0', '-ngld', 'all') }
if ($SpecType) { $a += @('--spec-type', $SpecType, '--spec-draft-n-max', "$NMax") }
$a += $Extra

$p = Start-Process -FilePath $Exe -ArgumentList $a -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logs "$Tag.out.log") -RedirectStandardError (Join-Path $logs "$Tag.err.log")
$p.Id | Set-Content $pidFile
$deadline = (Get-Date).AddSeconds(180)
while ((Get-Date) -lt $deadline) {
    try { if ((Invoke-RestMethod http://127.0.0.1:8191/health -TimeoutSec 3).status -eq 'ok') { break } } catch {}
    if ($p.HasExited) { throw "server exited with code $($p.ExitCode); see logs\$Tag.err.log" }
    Start-Sleep 3
}
Start-Sleep 2
# Windows can silently move CUDA memory to system RAM when VRAM is nearly full; that wrecks step times.
$s = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples |
     Where-Object { $_.InstanceName -match "pid_$($p.Id)" } | Measure-Object CookedValue -Sum
"READY pid=$($p.Id) shared_mb=$([math]::Round($s.Sum / 1MB, 1))"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
