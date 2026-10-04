$ErrorActionPreference = "Stop"

$projectDirectory = $PSScriptRoot
$python = Join-Path $projectDirectory ".venv\Scripts\python.exe"
$experimentRoot = Join-Path $projectDirectory "state_experiments"
$queueLog = Join-Path $experimentRoot "overnight-comparison.log"
$trainerPath = Join-Path $projectDirectory "state_q_network.py"
$budget = [TimeSpan]::FromHours(6)
$protocolId = "fair-state4-v1-500k-ls10000-tf4-eps0.1-0.01-200000"
$trainerHash = (Get-FileHash -LiteralPath $trainerPath -Algorithm SHA256).Hash
$queueStart = Get-Date
$pairDurationEstimates = [System.Collections.Generic.List[double]]::new()
$stoppedForBudget = $false

Set-Location $projectDirectory
New-Item -ItemType Directory -Force $experimentRoot | Out-Null

function Write-QueueLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $elapsed = (Get-Date) - $script:queueStart
    $line = "[{0}] [elapsed {1}] {2}" -f (
        (Get-Date -Format o), $elapsed.ToString("hh\:mm\:ss"), $Message)
    $line | Tee-Object -FilePath $script:queueLog -Append
}

function Get-ExpectedRun {
    param(
        [Parameter(Mandatory = $true)][string]$Algorithm,
        [Parameter(Mandatory = $true)][int]$Seed
    )

    $runName = "fair-$Algorithm-seed$Seed"
    [PSCustomObject]@{
        Algorithm = $Algorithm
        Seed = $Seed
        RunName = $runName
        RunDirectory = Join-Path $script:experimentRoot $runName
        MaxSteps = 500000
    }
}

function New-ProtocolRecord {
    param([Parameter(Mandatory = $true)]$Run)

    [ordered]@{
        protocol_id = $script:protocolId
        trainer_sha256 = $script:trainerHash
        algorithm = $Run.Algorithm
        seed = $Run.Seed
        representation = "state4"
        max_steps = $Run.MaxSteps
        episodes_limit = 50000
        hidden_size = 256
        batch_size = 128
        memory_size = 50000
        learning_starts = 10000
        train_frequency = 4
        learning_rate = 0.0001
        gamma = 0.99
        tau = 0.005
        epsilon_start = 0.1
        epsilon_end = 0.01
        epsilon_decay_steps = 200000
        fps = 0
    }
}

function Get-RunStatus {
    param([Parameter(Mandatory = $true)]$Run)

    if (-not (Test-Path -LiteralPath $Run.RunDirectory)) {
        return [PSCustomObject]@{
            TrainingComplete = $false
            EvaluationComplete = $false
        }
    }

    $protocolPath = Join-Path $Run.RunDirectory "protocol.json"
    $csvPath = Join-Path $Run.RunDirectory "episodes.csv"
    $evaluationPath = Join-Path $Run.RunDirectory "evaluation_seed999.csv"
    $partialInstruction = (
        "Move or delete the partial directory after preserving any files you need, " +
        "then run the queue again: {0}" -f $Run.RunDirectory)

    if (-not (Test-Path -LiteralPath $protocolPath)) {
        throw "Incomplete or unverified fair run '$($Run.RunName)': protocol.json is missing. $partialInstruction"
    }
    if (-not (Test-Path -LiteralPath $csvPath)) {
        $runConfigPath = Join-Path $Run.RunDirectory "run_config.json"
        $checkpointDirectory = Join-Path $Run.RunDirectory "checkpoints"
        $hasCheckpointFiles = (
            (Test-Path -LiteralPath $checkpointDirectory) -and
            @(Get-ChildItem -LiteralPath $checkpointDirectory -File -ErrorAction SilentlyContinue).Count -gt 0
        )
        if (
            -not (Test-Path -LiteralPath $runConfigPath) -and
            -not $hasCheckpointFiles
        ) {
            # The previous launcher may have stopped after writing only its
            # protocol file (for example, because PowerShell interpreted an
            # informational TensorFlow stderr line as an error). No training
            # state exists, so starting this run from scratch is safe.
            return [PSCustomObject]@{
                TrainingComplete = $false
                EvaluationComplete = $false
            }
        }
        throw "Incomplete fair run '$($Run.RunName)': episodes.csv is missing. $partialInstruction"
    }

    $protocol = Get-Content -LiteralPath $protocolPath -Raw | ConvertFrom-Json
    $expectedProtocol = New-ProtocolRecord -Run $Run
    $protocolMismatch = $false
    foreach ($name in $expectedProtocol.Keys) {
        $property = $protocol.PSObject.Properties[$name]
        if (
            $null -eq $property -or
            [string]$property.Value -ne [string]$expectedProtocol[$name]
        ) {
            $protocolMismatch = $true
            break
        }
    }
    if ($protocolMismatch) {
        throw "Protocol mismatch for '$($Run.RunName)'. Do not resume or mix this run with the final comparison. $partialInstruction"
    }

    $rows = @(Import-Csv -LiteralPath $csvPath)
    if ($rows.Count -eq 0) {
        throw "Incomplete fair run '$($Run.RunName)': episodes.csv has no data rows. $partialInstruction"
    }

    $previousEpisode = 0L
    $previousTimestep = 0L
    foreach ($row in $rows) {
        $episode = [int64]$row.episode
        $timestep = [int64]$row.timestep
        if (
            $row.algorithm -ne $Run.Algorithm -or
            $row.representation -ne "state4" -or
            [int]$row.seed -ne $Run.Seed -or
            $episode -le $previousEpisode -or
            $timestep -le $previousTimestep
        ) {
            throw "Invalid or mixed data in '$csvPath'. Do not use it for the final comparison. $partialInstruction"
        }
        $previousEpisode = $episode
        $previousTimestep = $timestep
    }

    $lastRow = $rows[-1]
    $lastTimestep = [int64]$lastRow.timestep
    $lastEpisode = [int64]$lastRow.episode
    if ($lastTimestep -ne $Run.MaxSteps) {
        throw "Incomplete fair run '$($Run.RunName)': found $lastTimestep of $($Run.MaxSteps) timesteps. Unsafe resume is disabled. $partialInstruction"
    }

    $checkpointIndex = Join-Path $Run.RunDirectory (
        "checkpoints\state-$($Run.Algorithm)-$lastEpisode.index")
    if (-not (Test-Path -LiteralPath $checkpointIndex)) {
        throw "Checkpoint/CSV mismatch for '$($Run.RunName)': expected '$checkpointIndex'. $partialInstruction"
    }

    $evaluationComplete = $false
    if (Test-Path -LiteralPath $evaluationPath) {
        $evaluationRows = @(Import-Csv -LiteralPath $evaluationPath)
        $invalidEvaluation = (
            $evaluationRows.Count -ne 50 -or
            @($evaluationRows | Where-Object {
                $_.algorithm -ne $Run.Algorithm -or
                $_.run_name -ne $Run.RunName -or
                [int]$_.evaluation_seed -ne 999
            }).Count -gt 0 -or
            (Get-Item -LiteralPath $evaluationPath).LastWriteTimeUtc -lt
                (Get-Item -LiteralPath $checkpointIndex).LastWriteTimeUtc
        )
        if ($invalidEvaluation) {
            throw "Invalid or stale evaluation for '$($Run.RunName)'. Remove only '$evaluationPath' and run the queue again."
        }
        $evaluationComplete = $true
    }

    return [PSCustomObject]@{
        TrainingComplete = $true
        EvaluationComplete = $evaluationComplete
    }
}

# Build the five paired seeds in advance and validate every existing fair run
# before starting any new training. A partial run aborts the entire queue.
$pairs = foreach ($seed in 42..46) {
    [PSCustomObject]@{
        Seed = $seed
        Runs = @(
            (Get-ExpectedRun -Algorithm "dqn" -Seed $seed),
            (Get-ExpectedRun -Algorithm "ddqn" -Seed $seed)
        )
    }
}

$statuses = @{}
foreach ($pair in $pairs) {
    foreach ($run in $pair.Runs) {
        $statuses[$run.RunName] = Get-RunStatus -Run $run
    }
}

Write-QueueLog "Fresh-run preflight passed for all five DQN/DDQN seed pairs."
Write-QueueLog "Soft operational budget: 6 hours. A started seed pair will always finish."

foreach ($pair in $pairs) {
    $elapsed = (Get-Date) - $queueStart
    $pairRequiresWork = @($pair.Runs | Where-Object {
        $status = $statuses[$_.RunName]
        -not $status.TrainingComplete -or -not $status.EvaluationComplete
    }).Count -gt 0
    if ($pairRequiresWork -and $pairDurationEstimates.Count -gt 0) {
        $estimatedPairSeconds = (
            $pairDurationEstimates | Measure-Object -Average).Average
        $projectedFinish = $elapsed.TotalSeconds + $estimatedPairSeconds
        Write-QueueLog (
            "Before seed {0}: elapsed {1}; estimated next-pair duration {2}." -f
            $pair.Seed, $elapsed.ToString("hh\:mm\:ss"),
            ([TimeSpan]::FromSeconds($estimatedPairSeconds)).ToString("hh\:mm\:ss"))
        if ($projectedFinish -gt $budget.TotalSeconds) {
            Write-QueueLog (
                (("Soft 6-hour budget would be exceeded by starting seed {0}; " +
                "stopping before this pair. Estimates are based on pairs " +
                "completed during this invocation.") -f $pair.Seed))
            $stoppedForBudget = $true
            break
        }
    }

    $pairStart = Get-Date
    $pairDidWork = $false
    $newTrainingRuns = 0
    Write-QueueLog "Starting paired seed $($pair.Seed)."

    # No budget check occurs inside this loop: once a seed pair starts, both
    # algorithms and their evaluations are allowed to finish.
    foreach ($run in $pair.Runs) {
        $status = $statuses[$run.RunName]
        if ($status.TrainingComplete) {
            Write-QueueLog "Skipping completed and validated training run $($run.RunName)."
        }
        else {
            $pairDidWork = $true
            $newTrainingRuns += 1
            New-Item -ItemType Directory -Force $run.RunDirectory | Out-Null
            $protocolPath = Join-Path $run.RunDirectory "protocol.json"
            New-ProtocolRecord -Run $run |
                ConvertTo-Json -Depth 3 |
                Set-Content -LiteralPath $protocolPath -Encoding UTF8

            Write-QueueLog "Training fresh run $($run.RunName) to $($run.MaxSteps) steps."
            $previousErrorActionPreference = $ErrorActionPreference
            $pythonExitCode = $null
            try {
                # TensorFlow writes normal startup information to stderr.
                # Keep it in the combined log, but judge the native process by
                # its exit code instead of PowerShell's NativeCommandError.
                $ErrorActionPreference = "Continue"
                & $python -u state_q_network.py `
                    --algorithm $run.Algorithm `
                    --run-name $run.RunName `
                    --from-scratch `
                    --seed $run.Seed `
                    --episodes 50000 `
                    --max-steps $run.MaxSteps `
                    --hidden-size 256 `
                    --batch-size 128 `
                    --memory-size 50000 `
                    --learning-starts 10000 `
                    --train-frequency 4 `
                    --learning-rate 0.0001 `
                    --gamma 0.99 `
                    --tau 0.005 `
                    --epsilon-start 0.1 `
                    --epsilon-end 0.01 `
                    --epsilon-decay-steps 200000 `
                    --checkpoint-interval 50 `
                    --fps 0 2>&1 |
                    Tee-Object -FilePath $queueLog -Append
                $pythonExitCode = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }

            if ($pythonExitCode -ne 0) {
                throw "$($run.RunName) failed with exit code $pythonExitCode. Its directory is now partial and must not be resumed."
            }
            $status = Get-RunStatus -Run $run
            $statuses[$run.RunName] = $status
        }

        if ($status.EvaluationComplete) {
            Write-QueueLog "Skipping completed and validated evaluation for $($run.RunName)."
        }
        else {
            $pairDidWork = $true
            Write-QueueLog "Evaluating $($run.RunName) for 50 greedy episodes."
            $previousErrorActionPreference = $ErrorActionPreference
            $pythonExitCode = $null
            try {
                $ErrorActionPreference = "Continue"
                & $python -u state_q_network.py `
                    --mode eval `
                    --algorithm $run.Algorithm `
                    --run-name $run.RunName `
                    --seed 999 `
                    --eval-episodes 50 2>&1 |
                    Tee-Object -FilePath $queueLog -Append
                $pythonExitCode = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }

            if ($pythonExitCode -ne 0) {
                throw "$($run.RunName) evaluation failed with exit code $pythonExitCode."
            }
            $status = Get-RunStatus -Run $run
            $statuses[$run.RunName] = $status
        }
    }

    $pairDuration = (Get-Date) - $pairStart
    Write-QueueLog (
        "Finished paired seed {0} in {1}." -f
        $pair.Seed, $pairDuration.ToString("hh\:mm\:ss"))
    if ($newTrainingRuns -gt 0) {
        # Scale a half-new pair to a full-pair equivalent when one validated
        # training run was already present. Evaluation time remains included.
        $fullPairEquivalent = (
            $pairDuration.TotalSeconds * 2.0 / $newTrainingRuns)
        $pairDurationEstimates.Add($fullPairEquivalent)
        Write-QueueLog (
            "Timing estimate added for seed {0}: {1} full-pair equivalent." -f
            $pair.Seed,
            ([TimeSpan]::FromSeconds($fullPairEquivalent)).ToString("hh\:mm\:ss"))
    }
    elseif ($pairDidWork) {
        Write-QueueLog (
            (("Seed {0} required evaluation only; its duration is not used to " +
            "project a full training pair.") -f $pair.Seed))
    }
}

$analysisCommand = (
    ".\.venv\Scripts\python.exe analyze_state_experiments.py " +
    "--run-prefix fair --seeds 42 43 44 45 46 --max-timestep 500000 " +
    "--output-dir state_experiments/fair-comparison-500k")
if ($stoppedForBudget) {
    $analysisCommand += " --allow-missing"
    Write-QueueLog "Queue stopped at a seed-pair boundary under the soft 6-hour budget."
}
else {
    Write-QueueLog "All available seed pairs and evaluations are complete."
}
Write-QueueLog "Analysis command: $analysisCommand"
Write-QueueLog "Total queue elapsed time: $(((Get-Date) - $queueStart).ToString('hh\:mm\:ss'))."
