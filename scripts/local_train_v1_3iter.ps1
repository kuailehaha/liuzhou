Param()

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repoRoot "logs"
$ckptDir = Join-Path $repoRoot "checkpoints_v1_local3_$timestamp"
$metricsPath = Join-Path $repoRoot "v1/data/stage_runs/local_3iter_metrics_$timestamp.json"
$logPath = Join-Path $logDir "local_train_v1_3iter_$timestamp.log"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
New-Item -ItemType Directory -Path $ckptDir -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $metricsPath) -Force | Out-Null

$env:PYTHONPATH = "./:./build/v0/src:./v0/build/src"
$env:CUDA_VISIBLE_DEVICES = "0"

$cmd = @(
    "run", "-n", "torchenv",
    "python", "scripts/train_entry.py",
    "--pipeline", "v1",
    "--stage", "all",
    "--iterations", "3",
    "--device", "cuda:0",
    "--devices", "cuda:0",
    "--train_devices", "cuda:0",
    "--infer_devices", "cuda:0",
    "--self_play_games", "64",
    "--mcts_simulations", "16",
    "--self_play_concurrent_games", "8",
    "--self_play_opening_random_moves", "8",
    "--max_game_plies", "160",
    "--batch_size", "64",
    "--epochs", "1",
    "--lr", "1e-3",
    "--weight_decay", "1e-4",
    "--soft_label_alpha", "1.0",
    "--temperature_init", "1.0",
    "--temperature_final", "0.1",
    "--temperature_threshold", "10",
    "--exploration_weight", "1.0",
    "--dirichlet_alpha", "0.3",
    "--dirichlet_epsilon", "0.25",
    "--soft_value_k", "2.0",
    "--checkpoint_dir", "$ckptDir",
    "--metrics_output", "$metricsPath"
)

Write-Host "[local_train_v1_3iter] log=$logPath"
Write-Host "[local_train_v1_3iter] checkpoint_dir=$ckptDir"
Write-Host "[local_train_v1_3iter] metrics_output=$metricsPath"

& conda @cmd 2>&1 | Tee-Object -FilePath $logPath
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "local_train_v1_3iter failed with exit code $exitCode. log=$logPath"
}

Write-Host "[local_train_v1_3iter] completed successfully."
Write-Host "[local_train_v1_3iter] log saved to $logPath"
