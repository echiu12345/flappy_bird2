param(
    [switch]$SkipEvaluation,
    [ValidatePattern("^[a-z0-9][a-z0-9-]*$")]
    [string]$RunPrefix = "long",
    [string]$OutputDirectoryName = "long-comparison-1.5m",
    [string]$QueueLogName = "long-horizon-comparison.log",
    [string[]]$LearningRateSchedule = @()
)

$ErrorActionPreference = "Stop"

$projectDirectory = $PSScriptRoot
$python = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$trainer = Join-Path $projectDirectory "state_q_network.py"
$analyzer = Join-Path $projectDirectory "analyze_evaluation_curves.py"
$experimentRoot = Join-Path $projectDirectory "state_experiments"
$outputDirectory = Join-Path $experimentRoot $OutputDirectoryName
$queueLog = Join-Path $experimentRoot $QueueLogName
$budget = [TimeSpan]::FromHours(6)
$queueStart = Get-Date
$trainingPairEstimateSeconds = 51 * 60
$evaluationPairEstimateSeconds = 12 * 60
$trainingPairDurations = [System.Collections.Generic.List[double]]::new()
$evaluationPairDurations = [System.Collections.Generic.List[double]]::new()
$selectedTimesteps = @(500000, 700000, 1000000, 1200000, 1500000)
$seeds = @(42, 43, 44, 45, 46)
$algorithms = @("dqn", "ddqn")
$maxSteps = 1500000
$checkpointInterval = 100000
$evaluationEpisodes = 100
$evaluationSeed = 999

Set-Location $projectDirectory
New-Item -ItemType Directory -Force $experimentRoot | Out-Null

function Write-QueueLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $elapsed = (Get-Date) - $script:queueStart
    $line = "[{0}] [elapsed {1}] {2}" -f (
        (Get-Date -Format o), $elapsed.ToString("hh\:mm\:ss"), $Message)
    $line | Tee-Object -FilePath $script:queueLog -Append
}

function Get-RunName {
    param([string]$Algorithm, [int]$Seed)
    return "$($script:RunPrefix)-$Algorithm-seed$Seed"
}

function Get-RunDirectory {
    param([string]$Algorithm, [int]$Seed)
    return Join-Path $script:experimentRoot (
        Get-RunName -Algorithm $Algorithm -Seed $Seed)
}

function Test-TrainingComplete {
    param([string]$Algorithm, [int]$Seed)

    $runDirectory = Get-RunDirectory -Algorithm $Algorithm -Seed $Seed
    $episodesPath = Join-Path $runDirectory "episodes.csv"
    if (-not (Test-Path -LiteralPath $episodesPath)) {
        return $false
    }
    $rows = @(Import-Csv -LiteralPath $episodesPath)
    if ($rows.Count -eq 0 -or [int64]$rows[-1].timestep -ne $script:maxSteps) {
        return $false
    }
    for ($timestep = $script:checkpointInterval;
         $timestep -le $script:maxSteps;
         $timestep += $script:checkpointInterval) {
        $indexPath = Join-Path $runDirectory (
            "step_checkpoints\state-$Algorithm-$timestep.index")
        if (-not (Test-Path -LiteralPath $indexPath)) {
            return $false
        }
    }
    return $true
}

function Test-EvaluationComplete {
    param([string]$Algorithm, [int]$Seed)

    $runDirectory = Get-RunDirectory -Algorithm $Algorithm -Seed $Seed
    $path = Join-Path $runDirectory (
        "evaluation_curve_seed$($script:evaluationSeed).csv")
    if (-not (Test-Path -LiteralPath $path)) {
        return $false
    }
    $rows = @(Import-Csv -LiteralPath $path)
    if ($rows.Count -ne (
            $script:selectedTimesteps.Count * $script:evaluationEpisodes)) {
        return $false
    }
    $recordedTimesteps = @(
        $rows | ForEach-Object { [int]$_.training_timestep } |
        Sort-Object -Unique)
    if (Compare-Object $recordedTimesteps $script:selectedTimesteps) {
        return $false
    }
    return $true
}

function Invoke-LoggedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $previousErrorActionPreference = $ErrorActionPreference
    $exitCode = $null
    try {
        # TensorFlow writes ordinary startup information to stderr.
        $ErrorActionPreference = "Continue"
        & $script:python -u @Arguments 2>&1 |
            Tee-Object -FilePath $script:queueLog -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code $exitCode. See $($script:queueLog)."
    }
}

# Fresh-run preflight: do not mix previous or partial long-horizon data.
foreach ($seed in $seeds) {
    foreach ($algorithm in $algorithms) {
        $runDirectory = Get-RunDirectory -Algorithm $algorithm -Seed $seed
        if (Test-Path -LiteralPath $runDirectory) {
            throw "Long-horizon run directory already exists: $runDirectory. Preserve and move partial data before starting a fresh experiment."
        }
    }
}

"" | Set-Content -LiteralPath $queueLog -Encoding UTF8
Write-QueueLog "Fresh preflight passed for five DQN/DDQN seed pairs."
Write-QueueLog "Operational budget is six hours; a started seed pair always finishes."
Write-QueueLog "Run prefix: $RunPrefix. Learning-rate schedule: $($LearningRateSchedule -join ', ')."

$completedSeeds = [System.Collections.Generic.List[int]]::new()
foreach ($seed in $seeds) {
    $elapsedSeconds = ((Get-Date) - $queueStart).TotalSeconds
    $estimatedSeconds = if ($trainingPairDurations.Count) {
        ($trainingPairDurations | Measure-Object -Average).Average
    }
    else {
        $trainingPairEstimateSeconds
    }
    if ($elapsedSeconds + $estimatedSeconds -gt $budget.TotalSeconds) {
        Write-QueueLog "Stopping before seed $seed because the estimated pair would exceed the six-hour budget."
        break
    }

    $pairStart = Get-Date
    Write-QueueLog "Starting paired training seed $seed."
    foreach ($algorithm in $algorithms) {
        $runName = Get-RunName -Algorithm $algorithm -Seed $seed
        Write-QueueLog "Training $runName to $maxSteps timesteps."
        $arguments = @(
            "state_q_network.py",
            "--algorithm", $algorithm,
            "--run-name", $runName,
            "--from-scratch",
            "--seed", [string]$seed,
            "--episodes", "100000",
            "--max-steps", [string]$maxSteps,
            "--hidden-size", "256",
            "--batch-size", "128",
            "--memory-size", "50000",
            "--learning-starts", "10000",
            "--train-frequency", "4",
            "--learning-rate", "0.0001",
            "--gamma", "0.99",
            "--tau", "0.005",
            "--epsilon-start", "0.1",
            "--epsilon-end", "0.01",
            "--epsilon-decay-steps", "200000",
            "--checkpoint-interval", "50",
            "--step-checkpoint-interval", [string]$checkpointInterval,
            "--fps", "0"
        )
        if ($LearningRateSchedule.Count -gt 0) {
            $arguments += "--learning-rate-schedule"
            $arguments += $LearningRateSchedule
        }
        Invoke-LoggedPython -Arguments $arguments
        if (-not (Test-TrainingComplete -Algorithm $algorithm -Seed $seed)) {
            throw "$runName did not pass post-training validation."
        }
    }
    $duration = ((Get-Date) - $pairStart).TotalSeconds
    $trainingPairDurations.Add($duration)
    $completedSeeds.Add($seed)
    Write-QueueLog "Finished paired training seed $seed in $([TimeSpan]::FromSeconds($duration).ToString('hh\:mm\:ss'))."
}

if ($SkipEvaluation) {
    Write-QueueLog "Training phase complete; evaluation skipped by request."
    exit 0
}

foreach ($seed in $completedSeeds) {
    $elapsedSeconds = ((Get-Date) - $queueStart).TotalSeconds
    $estimatedSeconds = if ($evaluationPairDurations.Count) {
        ($evaluationPairDurations | Measure-Object -Average).Average
    }
    else {
        $evaluationPairEstimateSeconds
    }
    if ($elapsedSeconds + $estimatedSeconds -gt $budget.TotalSeconds) {
        Write-QueueLog "Stopping evaluation before seed $seed to respect the six-hour soft budget."
        break
    }

    $pairStart = Get-Date
    foreach ($algorithm in $algorithms) {
        $runName = Get-RunName -Algorithm $algorithm -Seed $seed
        Write-QueueLog "Evaluating selected checkpoints for $runName."
        $arguments = @(
            "state_q_network.py",
            "--mode", "eval-curve",
            "--algorithm", $algorithm,
            "--run-name", $runName,
            "--seed", [string]$evaluationSeed,
            "--max-steps", [string]$maxSteps,
            "--step-checkpoint-interval", [string]$checkpointInterval,
            "--eval-timesteps"
        ) + @($selectedTimesteps | ForEach-Object { [string]$_ }) + @(
            "--eval-episodes", [string]$evaluationEpisodes,
            "--fps", "0"
        )
        Invoke-LoggedPython -Arguments $arguments
        if (-not (Test-EvaluationComplete -Algorithm $algorithm -Seed $seed)) {
            throw "$runName did not pass post-evaluation validation."
        }
    }
    $duration = ((Get-Date) - $pairStart).TotalSeconds
    $evaluationPairDurations.Add($duration)
    Write-QueueLog "Finished paired evaluation seed $seed in $([TimeSpan]::FromSeconds($duration).ToString('hh\:mm\:ss'))."
}

$fullyEvaluatedSeeds = @(
    foreach ($seed in $completedSeeds) {
        if ((Test-EvaluationComplete -Algorithm "dqn" -Seed $seed) -and
            (Test-EvaluationComplete -Algorithm "ddqn" -Seed $seed)) {
            $seed
        }
    }
)

if ($fullyEvaluatedSeeds.Count -eq $seeds.Count) {
    Write-QueueLog "All five paired seeds are complete; generating long-horizon analysis."
    $arguments = @(
        "analyze_evaluation_curves.py",
        "--run-prefix", $RunPrefix,
        "--seeds"
    ) + @($seeds | ForEach-Object { [string]$_ }) + @(
        "--timesteps"
    ) + @($selectedTimesteps | ForEach-Object { [string]$_ }) + @(
        "--episodes-per-checkpoint", [string]$evaluationEpisodes,
        "--output-dir", $outputDirectory
    )
    Invoke-LoggedPython -Arguments $arguments
    Write-QueueLog "Analysis complete: $outputDirectory"
}
else {
    Write-QueueLog "Evaluation is incomplete for the full five-seed design; analysis was not generated."
}

Write-QueueLog "Total elapsed time: $(((Get-Date) - $queueStart).ToString('hh\:mm\:ss'))."
