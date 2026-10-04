$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "run_long_horizon_comparison.ps1") `
    -RunPrefix "lrdecay" `
    -OutputDirectoryName "lrdecay-comparison-1.5m" `
    -QueueLogName "lrdecay-comparison.log" `
    -LearningRateSchedule @(
        "500000:0.00005",
        "1000000:0.00001"
    )

& (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") `
    (Join-Path $PSScriptRoot "analyze_lr_decay_effect.py") `
    --baseline-prefix long `
    --treatment-prefix lrdecay `
    --seeds 42 43 44 45 46 `
    --timesteps 500000 700000 1000000 1200000 1500000 `
    --episodes-per-checkpoint 100 `
    --output-dir state_experiments/lrdecay-vs-fixed-1.5m

if ($LASTEXITCODE -ne 0) {
    throw "Learning-rate decay comparison analysis failed with exit code $LASTEXITCODE."
}
