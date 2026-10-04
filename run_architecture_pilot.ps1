param(
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"

$projectDirectory = $PSScriptRoot
$python = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$experimentRoot = Join-Path $projectDirectory "state_experiments"
$queueLog = Join-Path $experimentRoot "architecture-pilot.log"
$outputDirectory = Join-Path $experimentRoot "architecture-pilot-500k"
$seeds = @(42, 43, 44)
$algorithms = @("dqn", "ddqn")
$architectures = @(
    @{ Prefix = "arch256"; HiddenSize = 256; Label = "256x256" },
    @{ Prefix = "arch128"; HiddenSize = 128; Label = "128x128" }
)
$maxSteps = 500000
$checkpointInterval = 100000
$selectedTimesteps = @(100000, 200000, 300000, 400000, 500000)
$evaluationEpisodes = 50
$evaluationSeed = 999
$queueStart = Get-Date

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
    param([string]$Prefix, [string]$Algorithm, [int]$Seed)
    return "$Prefix-$Algorithm-seed$Seed"
}

function Get-RunDirectory {
    param([string]$Prefix, [string]$Algorithm, [int]$Seed)
    return Join-Path $script:experimentRoot (
        Get-RunName -Prefix $Prefix -Algorithm $Algorithm -Seed $Seed)
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

function Test-TrainingComplete {
    param([string]$Prefix, [string]$Algorithm, [int]$Seed)

    $runDirectory = Get-RunDirectory -Prefix $Prefix -Algorithm $Algorithm -Seed $Seed
    $episodesPath = Join-Path $runDirectory "episodes.csv"
    if (-not (Test-Path -LiteralPath $episodesPath)) {
        return $false
    }
    $rows = @(Import-Csv -LiteralPath $episodesPath)
    if ($rows.Count -eq 0 -or [int64]$rows[-1].timestep -ne $script:maxSteps) {
        return $false
    }
    foreach ($timestep in $script:selectedTimesteps) {
        $indexPath = Join-Path $runDirectory (
            "step_checkpoints\state-$Algorithm-$timestep.index")
        if (-not (Test-Path -LiteralPath $indexPath)) {
            return $false
        }
    }
    return $true
}

function Test-EvaluationComplete {
    param([string]$Prefix, [string]$Algorithm, [int]$Seed)

    $runDirectory = Get-RunDirectory -Prefix $Prefix -Algorithm $Algorithm -Seed $Seed
    $path = Join-Path $runDirectory (
        "evaluation_curve_seed$($script:evaluationSeed).csv")
    if (-not (Test-Path -LiteralPath $path)) {
        return $false
    }
    $rows = @(Import-Csv -LiteralPath $path)
    if ($rows.Count -ne ($script:selectedTimesteps.Count * $script:evaluationEpisodes)) {
        return $false
    }
    $recordedTimesteps = @(
        $rows | ForEach-Object { [int]$_.training_timestep } |
        Sort-Object -Unique)
    return -not [bool](Compare-Object $recordedTimesteps $script:selectedTimesteps)
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual-environment Python was not found: $python"
}

# Do not mix prior or partial data into a fresh pilot.
foreach ($architecture in $architectures) {
    foreach ($seed in $seeds) {
        foreach ($algorithm in $algorithms) {
            $runDirectory = Get-RunDirectory `
                -Prefix $architecture.Prefix -Algorithm $algorithm -Seed $seed
            if (Test-Path -LiteralPath $runDirectory) {
                throw "Pilot run directory already exists: $runDirectory. Preserve and move it before starting a fresh pilot."
            }
        }
    }
}

"" | Set-Content -LiteralPath $queueLog -Encoding UTF8
Write-QueueLog "Fresh preflight passed for 12 runs: two architectures, two algorithms, and three paired seeds."
Write-QueueLog "Pilot settings: 500000 steps, checkpoints every 100000 steps, 50 greedy episodes per checkpoint."

foreach ($seed in $seeds) {
    foreach ($architecture in $architectures) {
        foreach ($algorithm in $algorithms) {
            $runName = Get-RunName `
                -Prefix $architecture.Prefix -Algorithm $algorithm -Seed $seed
            Write-QueueLog "Training $runName ($($architecture.Label)) to $maxSteps timesteps."
            $arguments = @(
                "state_q_network.py",
                "--algorithm", $algorithm,
                "--run-name", $runName,
                "--from-scratch",
                "--seed", [string]$seed,
                "--episodes", "100000",
                "--max-steps", [string]$maxSteps,
                "--hidden-size", [string]$architecture.HiddenSize,
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
            Invoke-LoggedPython -Arguments $arguments
            if (-not (Test-TrainingComplete `
                    -Prefix $architecture.Prefix -Algorithm $algorithm -Seed $seed)) {
                throw "$runName did not pass post-training validation."
            }
        }
    }
}

if ($SkipEvaluation) {
    Write-QueueLog "Training phase complete; evaluation skipped by request."
    exit 0
}

foreach ($seed in $seeds) {
    foreach ($architecture in $architectures) {
        foreach ($algorithm in $algorithms) {
            $runName = Get-RunName `
                -Prefix $architecture.Prefix -Algorithm $algorithm -Seed $seed
            Write-QueueLog "Evaluating checkpoints for $runName."
            $arguments = @(
                "state_q_network.py",
                "--mode", "eval-curve",
                "--algorithm", $algorithm,
                "--run-name", $runName,
                "--seed", [string]$evaluationSeed,
                "--max-steps", [string]$maxSteps,
                "--hidden-size", [string]$architecture.HiddenSize,
                "--step-checkpoint-interval", [string]$checkpointInterval,
                "--eval-timesteps"
            ) + @($selectedTimesteps | ForEach-Object { [string]$_ }) + @(
                "--eval-episodes", [string]$evaluationEpisodes,
                "--fps", "0"
            )
            Invoke-LoggedPython -Arguments $arguments
            if (-not (Test-EvaluationComplete `
                    -Prefix $architecture.Prefix -Algorithm $algorithm -Seed $seed)) {
                throw "$runName did not pass post-evaluation validation."
            }
        }
    }
}

Write-QueueLog "All pilot runs and evaluations are complete; generating architecture analysis."
$arguments = @(
    "analyze_architecture_effect.py",
    "--seeds"
) + @($seeds | ForEach-Object { [string]$_ }) + @(
    "--timesteps"
) + @($selectedTimesteps | ForEach-Object { [string]$_ }) + @(
    "--episodes-per-checkpoint", [string]$evaluationEpisodes,
    "--output-dir", $outputDirectory
)
Invoke-LoggedPython -Arguments $arguments
Write-QueueLog "Analysis complete: $outputDirectory"
Write-QueueLog "Total elapsed time: $(((Get-Date) - $queueStart).ToString('hh\:mm\:ss'))."

