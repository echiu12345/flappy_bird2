# Flappy Bird DQN / Double DQN experiment plan

## Recommended topic

**基於低維狀態輸入之 DQN 與 Double DQN 在 Flappy Bird 環境中的樣本效率與訓練穩定性比較**

This title matches the implemented scope: two value-based algorithms receive
the same four numeric state features. “Training stability” is more precise than
claiming theoretical convergence. Q-value overestimation should not appear in
the title unless predicted values are compared with empirical discounted
returns from fixed states.

## Research questions

1. With the same 500,000-environment-step budget, which algorithm has greater
   sample efficiency, measured by the area under its score learning curve?
2. Which algorithm reaches fixed score criteria sooner?
3. Which algorithm has higher late-stage and held-out greedy-policy scores?
4. Which algorithm has lower cross-seed variation and less degradation after
   its peak performance?

## Controlled experiment

- Algorithms: DQN and Double DQN.
- State: normalized bird height, vertical velocity, next-pipe horizontal
  distance, and signed distance to the next gap center.
- Paired training seeds: 42, 43, 44, 45, and 46.
- Budget: 500,000 environment steps for every run.
- Replay warm-up: collect 10,000 transitions before gradient updates.
- Epsilon: hold 0.1 during warm-up, linearly anneal 0.1 to 0.01 over
  the next 200,000 timesteps, then hold 0.01.
- Update frequency: one gradient update every four environment steps.
- All other network, optimizer, replay, reward, target-update, and environment
  settings are identical.
- Runs execute sequentially on the same computer without rendering.
- Final evaluation: 50 greedy episodes (`epsilon = 0`) with held-out
  evaluation seed 999.
- Every final run starts from random weights. The queue never resumes an
  incomplete `fair-` run because the current checkpoints do not include replay
  memory or random-number-generator state.
- The queue stores `protocol.json`, including the fixed queue parameters and a
  hash of the training script. The trainer independently stores
  `run_config.json` with the scientific hyperparameters. Only a complete
  500,000-step CSV with its matching final checkpoint and protocol may be
  reused or skipped.

Seeds and thresholds are chosen before inspecting the final comparison. The
seed numbers themselves have no special meaning; pairing ensures both
algorithms are tested with the same predefined sources of randomness.

The 500,000-step budget is an experimental design choice rather than a known
convergence point. The pilot DQN reached its best 100-episode average near
241,405 steps and later degraded, so 500,000 steps covers initial learning,
peak performance, and a substantial post-peak stability interval while still
making five paired runs practical.

## Six-hour operating policy

Six hours is a soft scheduling budget, not part of the statistical protocol
and not a guaranteed completion time. Runs remain paired by seed. Before
starting another seed pair, the queue compares elapsed time plus the average
duration of pairs completed during the current invocation with six hours. It
stops at the pair boundary when the projection exceeds the budget. The first
pair starts without an estimate, and a pair that has already started always
finishes both algorithms and both 50-episode evaluations; actual elapsed time
can therefore exceed six hours.

If the budget stops the queue before all five pairs, the completed pairs form
an interim descriptive result. Running the queue again starts the next missing
pair while skipping completed runs that pass protocol, CSV, and checkpoint
validation. A partial run aborts preflight and must be moved or deleted rather
than resumed.

## Primary figures and metrics

1. **Learning curve:** score against environment timestep. At every 10,000
   steps, average episodes completed during the preceding 50,000 steps. Show
   every seed faintly and the algorithm mean with ±1 sample standard deviation.
2. **Greedy evaluation:** connect DQN and Double DQN mean evaluation scores for
   each paired training seed. Treat the training seed, not each episode, as the
   independent replicate.
3. **Sample efficiency:** timestep-normalized area under the fixed-window score
   curve.
4. **Training stabilization:** first sustained score threshold, mean and
   variability during the final 20% of training, and peak-to-late degradation.
5. **Diagnostics:** predicted maximum Q and Huber loss. Loss is an optimization
   diagnostic, not a direct measure of game performance.

Predicted Q values alone cannot prove overestimation because different policies
visit different states and the true action values are unknown. A rising Q curve
with a falling score is evidence of value/performance divergence. A direct
overestimation study would additionally compare Q predictions on fixed states
with empirical discounted returns from repeated rollouts.

## Fixed-checkpoint extension

To measure policy learning dynamics directly, a second controlled dataset uses
`curve-` run names and the same five paired training seeds and hyperparameters.
It saves checkpoints at 50,000, 100,000, ..., 500,000 timesteps. Each checkpoint
is evaluated greedily on the same 20 episode seeds (999 through 1018). The
original `fair-` data remains the preliminary final-policy experiment; it is not
overwritten or mixed with the new fixed-checkpoint curves.

Run the fixed-checkpoint queue:

```powershell
.\run_checkpoint_comparison.ps1
```

## Commands

Run the complete local queue:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_overnight_comparison.ps1
```

Generate all final figures, CSV summaries, and the Markdown report:

```powershell
.\.venv\Scripts\python.exe analyze_state_experiments.py --run-prefix fair --seeds 42 43 44 45 46 --max-timestep 500000 --output-dir state_experiments/fair-comparison-500k
```

The queue logs ISO timestamps, elapsed time, completed pair durations, its
budget decision, total runtime, and the appropriate analysis command in
`state_experiments/overnight-comparison.log`. If the soft budget leaves some
pairs unfinished, use the same command with `--allow-missing` for an interim
report; do not present that interim report as the planned five-seed final
comparison.

The earlier `state-dqn-seed42` data used episode-based epsilon decay. Keep it as
pilot evidence and do not mix it into the final paired analysis.
