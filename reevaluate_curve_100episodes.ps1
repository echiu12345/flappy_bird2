$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$root = "state_experiments"
$log = Join-Path $root "reevaluation-100episodes.log"
$algorithms = @("dqn", "ddqn")
$seeds = 42..46

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

foreach ($algorithm in $algorithms) {
    foreach ($trainingSeed in $seeds) {
        $runName = "curve-$algorithm-seed$trainingSeed"
        $runDirectory = Join-Path $root $runName
        $source = Join-Path $runDirectory "evaluation_curve_seed999.csv"
        $backup = Join-Path $runDirectory "evaluation_curve_seed999_20episodes.csv"
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Existing 20-episode evaluation is missing: $source"
        }
        $rowCount = @(Import-Csv -LiteralPath $source).Count
        if ($rowCount -eq 200 -and -not (Test-Path -LiteralPath $backup)) {
            Copy-Item -LiteralPath $source -Destination $backup
        }
        elseif ($rowCount -ne 200 -and $rowCount -ne 1000) {
            throw "Unexpected row count $rowCount in $source"
        }
    }
}

"[$(Get-Date -Format o)] Starting 100-episode checkpoint reevaluation." |
    Set-Content -LiteralPath $log -Encoding UTF8

foreach ($algorithm in $algorithms) {
    foreach ($trainingSeed in $seeds) {
        $runName = "curve-$algorithm-seed$trainingSeed"
        Write-Host "[$(Get-Date -Format T)] Evaluating $runName (10 checkpoints x 100 episodes)"
        & $python -u state_q_network.py `
            --mode eval-curve `
            --algorithm $algorithm `
            --run-name $runName `
            --seed 999 `
            --max-steps 500000 `
            --step-checkpoint-interval 50000 `
            --eval-episodes 100 *>> $log
        if ($LASTEXITCODE -ne 0) {
            throw "$runName evaluation failed with exit code $LASTEXITCODE. See $log"
        }
        $output = Join-Path (Join-Path $root $runName) "evaluation_curve_seed999.csv"
        $rowCount = @(Import-Csv -LiteralPath $output).Count
        if ($rowCount -ne 1000) {
            throw "$runName produced $rowCount rows; expected 1000."
        }
    }
}

& $python analyze_evaluation_curves.py `
    --run-prefix curve `
    --seeds 42 43 44 45 46 `
    --max-timestep 500000 `
    --checkpoint-interval 50000 `
    --episodes-per-checkpoint 100 `
    --output-dir state_experiments/curve-comparison-500k *>> $log
if ($LASTEXITCODE -ne 0) {
    throw "Analysis failed with exit code $LASTEXITCODE. See $log"
}

Write-Host "[$(Get-Date -Format T)] Reevaluation and 95% CI analysis complete."
Write-Host "Results: state_experiments\curve-comparison-500k"
