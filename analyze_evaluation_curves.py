#!/usr/bin/env python
"""Plot fixed-checkpoint greedy evaluation curves for state DQN experiments."""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("dqn", "ddqn")
LABELS = {"dqn": "DQN", "ddqn": "Double DQN"}
COLORS = {"dqn": "tab:blue", "ddqn": "tab:orange"}


def bootstrap_mean_ci(scores, seed, samples=10000):
    """Return a deterministic percentile-bootstrap 95% CI for one policy."""
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scores), size=(samples, len(scores)))
    means = np.mean(scores[indices], axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def t_mean_ci(values):
    """Return a two-sided 95% t interval for independent training seeds."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        value = float(values[0])
        return value, value
    # Exact two-sided 0.975 quantiles for the seed counts used by this study;
    # the normal fallback is only used for unexpectedly large samples.
    critical = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        24: 2.064, 29: 2.045, 39: 2.023, 59: 2.001,
    }.get(len(values) - 1, 1.96)
    mean = float(np.mean(values))
    half_width = critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def parse_int(value, field, path):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s has invalid %s=%r" % (path, field, value))
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError("%s has non-integer %s=%r" % (path, field, value))
    return int(parsed)


def parse_bool(value, field, path):
    value = str(value).strip().lower()
    if value in ("0", "false", "no", ""):
        return False
    if value in ("1", "true", "yes"):
        return True
    raise ValueError("%s has invalid %s=%r" % (path, field, value))


def load_curve(path, algorithm, run_name, evaluation_seed,
               expected_timesteps, episodes_per_checkpoint):
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    required = {
        "algorithm", "run_name", "training_timestep", "evaluation_seed",
        "eval_episode_seed", "episode", "score", "frames", "truncated",
    }
    missing = required.difference(fields)
    if missing:
        raise ValueError(
            "%s is missing columns: %s" %
            (path, ", ".join(sorted(missing))))
    expected_rows = len(expected_timesteps) * episodes_per_checkpoint
    if len(rows) != expected_rows:
        raise ValueError(
            "%s has %d rows; expected %d" %
            (path, len(rows), expected_rows))
    if any(row["algorithm"].lower() != algorithm for row in rows):
        raise ValueError("Incorrect algorithm in " + str(path))
    if any(row["run_name"] != run_name for row in rows):
        raise ValueError("Incorrect run_name in " + str(path))
    if any(parse_int(row["evaluation_seed"], "evaluation_seed", path)
           != evaluation_seed for row in rows):
        raise ValueError("Incorrect evaluation seed in " + str(path))

    grouped = defaultdict(list)
    for row in rows:
        timestep = parse_int(
            row["training_timestep"], "training_timestep", path)
        episode = parse_int(row["episode"], "episode", path)
        eval_episode_seed = parse_int(
            row["eval_episode_seed"], "eval_episode_seed", path)
        score = parse_int(row["score"], "score", path)
        frames = parse_int(row["frames"], "frames", path)
        truncated = parse_bool(row["truncated"], "truncated", path)
        if score < 0 or frames <= 0:
            raise ValueError("Negative score or non-positive frames in " + str(path))
        grouped[timestep].append({
            "episode": episode,
            "eval_episode_seed": eval_episode_seed,
            "score": score,
            "frames": frames,
            "truncated": truncated,
        })

    if sorted(grouped) != expected_timesteps:
        raise ValueError(
            "%s contains timesteps %s; expected %s" %
            (path, sorted(grouped), expected_timesteps))
    expected_eval_seeds = list(range(
        evaluation_seed, evaluation_seed + episodes_per_checkpoint))
    points = []
    for timestep in expected_timesteps:
        checkpoint_rows = grouped[timestep]
        episodes = [row["episode"] for row in checkpoint_rows]
        eval_seeds = [row["eval_episode_seed"] for row in checkpoint_rows]
        if episodes != list(range(1, episodes_per_checkpoint + 1)):
            raise ValueError(
                "%s timestep %d has incorrect episode order" %
                (path, timestep))
        if eval_seeds != expected_eval_seeds:
            raise ValueError(
                "%s timestep %d does not use the shared evaluation seeds" %
                (path, timestep))
        scores = np.asarray(
            [row["score"] for row in checkpoint_rows], dtype=np.float64)
        bootstrap_seed = (
            20261003 + timestep +
            sum((index + 1) * ord(char)
                for index, char in enumerate(run_name + algorithm)))
        ci_low, ci_high = bootstrap_mean_ci(scores, bootstrap_seed)
        points.append({
            "training_timestep": timestep,
            "evaluation_score_mean": float(np.mean(scores)),
            "evaluation_score_ci95_low": ci_low,
            "evaluation_score_ci95_high": ci_high,
            "evaluation_score_median": float(np.median(scores)),
            "evaluation_score_std": float(np.std(scores, ddof=0)),
            "evaluation_score_max": int(np.max(scores)),
            "truncated_episodes": sum(
                row["truncated"] for row in checkpoint_rows),
        })
    return points


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(per_run_rows, timesteps):
    rows = []
    for algorithm in ALGORITHMS:
        for timestep in timesteps:
            values = np.asarray([
                row["evaluation_score_mean"] for row in per_run_rows
                if row["algorithm"] == algorithm
                and row["training_timestep"] == timestep
            ], dtype=np.float64)
            ci_low, ci_high = t_mean_ci(values)
            rows.append({
                "algorithm": algorithm,
                "training_timestep": timestep,
                "training_seeds": len(values),
                "mean_of_seed_means": float(np.mean(values)),
                "mean_ci95_low": ci_low,
                "mean_ci95_high": ci_high,
                "seed_mean_std": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                "median_of_seed_means": float(np.median(values)),
                "min_seed_mean": float(np.min(values)),
                "max_seed_mean": float(np.max(values)),
            })
    return rows


def summarize_paired_differences(per_run_rows, timesteps, seeds):
    rows = []
    for timestep in timesteps:
        by_key = {
            (row["algorithm"], row["seed"]): row["evaluation_score_mean"]
            for row in per_run_rows
            if row["training_timestep"] == timestep
        }
        dqn = np.asarray([by_key[("dqn", seed)] for seed in seeds])
        ddqn = np.asarray([by_key[("ddqn", seed)] for seed in seeds])
        differences = dqn - ddqn
        ci_low, ci_high = t_mean_ci(differences)
        rows.append({
            "training_timestep": timestep,
            "paired_training_seeds": len(seeds),
            "dqn_mean": float(np.mean(dqn)),
            "ddqn_mean": float(np.mean(ddqn)),
            "mean_difference_dqn_minus_ddqn": float(np.mean(differences)),
            "difference_ci95_low": ci_low,
            "difference_ci95_high": ci_high,
            "ci_excludes_zero": int(ci_low > 0 or ci_high < 0),
        })
    return rows


def plot_curves(path, per_run_rows, summary_rows, timesteps, seeds,
                episodes_per_checkpoint):
    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    for algorithm in ALGORITHMS:
        for seed in seeds:
            selected = sorted([
                row for row in per_run_rows
                if row["algorithm"] == algorithm and row["seed"] == seed
            ], key=lambda row: row["training_timestep"])
            axis.plot(
                [row["training_timestep"] for row in selected],
                [row["evaluation_score_mean"] for row in selected],
                color=COLORS[algorithm], alpha=0.18, linewidth=1.1)
        selected_summary = sorted([
            row for row in summary_rows if row["algorithm"] == algorithm
        ], key=lambda row: row["training_timestep"])
        x = np.asarray([
            row["training_timestep"] for row in selected_summary],
            dtype=np.float64)
        mean = np.asarray([
            row["mean_of_seed_means"] for row in selected_summary],
            dtype=np.float64)
        low = np.asarray([
            row["mean_ci95_low"] for row in selected_summary],
            dtype=np.float64)
        high = np.asarray([
            row["mean_ci95_high"] for row in selected_summary],
            dtype=np.float64)
        axis.fill_between(
            x, low, high,
            color=COLORS[algorithm], alpha=0.16)
        axis.plot(
            x, mean, color=COLORS[algorithm], linewidth=2.6,
            marker="o", markersize=5,
            label="%s mean with 95%% CI (n=%d seeds)" %
                  (LABELS[algorithm], len(seeds)))
    axis.set_title(
        "Greedy policy performance across training\n"
        "%d fixed evaluation episodes per checkpoint; faint lines are seeds" %
        episodes_per_checkpoint,
        fontsize=14)
    axis.set_xlabel("Training timestep")
    axis.set_ylabel("Mean greedy evaluation score")
    axis.set_xticks(timesteps)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path, summary_rows, seeds, timesteps,
                 episodes_per_checkpoint, evaluation_seed):
    lines = [
        "# Fixed-checkpoint greedy evaluation\n\n",
        "- Training seeds: **%s**.\n" %
        ", ".join(str(seed) for seed in seeds),
        "- Checkpoints: **%s** timesteps.\n" %
        ", ".join(format(value, ",") for value in timesteps),
        "- Evaluation episodes per checkpoint: **%d**.\n" %
        episodes_per_checkpoint,
        "- Shared evaluation seeds: **%d–%d**.\n\n" % (
            evaluation_seed,
            evaluation_seed + episodes_per_checkpoint - 1),
        "Each checkpoint is evaluated with `epsilon = 0`. The independent "
        "replicates for algorithm comparison are the training seeds; the "
        "episodes within a checkpoint estimate each trained policy's score.\n\n",
        "| Algorithm | Timestep | Mean greedy score | 95% CI | Across-seed SD | Range |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary_rows:
        lines.append(
            "| %s | %s | %.2f | %.2f–%.2f | %.2f | %.2f–%.2f |\n" % (
                LABELS[row["algorithm"]],
                format(row["training_timestep"], ","),
                row["mean_of_seed_means"], row["mean_ci95_low"],
                row["mean_ci95_high"], row["seed_mean_std"],
                row["min_seed_mean"], row["max_seed_mean"]))
    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze fixed-checkpoint greedy evaluation curves.")
    parser.add_argument("--root", default="state_experiments")
    parser.add_argument("--run-prefix", default="curve")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-timestep", type=int, default=500000)
    parser.add_argument("--checkpoint-interval", type=int, default=50000)
    parser.add_argument(
        "--timesteps", nargs="+", type=int,
        help=("analyze these explicit training timesteps instead of the "
              "regular checkpoint interval"))
    parser.add_argument("--evaluation-seed", type=int, default=999)
    parser.add_argument("--episodes-per-checkpoint", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="state_experiments/curve-comparison-500k")
    args = parser.parse_args()
    if args.timesteps:
        if any(timestep <= 0 for timestep in args.timesteps):
            parser.error("--timesteps values must be positive")
        if args.timesteps != sorted(set(args.timesteps)):
            parser.error("--timesteps must be unique and strictly increasing")
        timesteps = list(args.timesteps)
    else:
        if args.max_timestep <= 0 or args.checkpoint_interval <= 0:
            parser.error("timestep values must be positive")
        if args.max_timestep % args.checkpoint_interval:
            parser.error(
                "--max-timestep must be divisible by checkpoint interval")
        timesteps = list(range(
            args.checkpoint_interval,
            args.max_timestep + 1,
            args.checkpoint_interval))
    if args.episodes_per_checkpoint <= 0:
        parser.error("--episodes-per-checkpoint must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")

    root = Path(args.root)
    per_run_rows = []
    for algorithm in ALGORITHMS:
        for seed in args.seeds:
            run_name = "%s-%s-seed%d" % (
                args.run_prefix, algorithm, seed)
            path = root / run_name / (
                "evaluation_curve_seed%d.csv" % args.evaluation_seed)
            if not path.is_file():
                raise FileNotFoundError(str(path))
            points = load_curve(
                path, algorithm, run_name, args.evaluation_seed,
                timesteps, args.episodes_per_checkpoint)
            for point in points:
                point.update({"algorithm": algorithm, "seed": seed})
                per_run_rows.append(point)

    summary_rows = summarize(per_run_rows, timesteps)
    paired_rows = summarize_paired_differences(
        per_run_rows, timesteps, args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_curve_per_run.csv", per_run_rows)
    write_csv(output_dir / "evaluation_curve_summary.csv", summary_rows)
    write_csv(
        output_dir / "evaluation_curve_paired_differences.csv", paired_rows)
    plot_curves(
        output_dir / "greedy_evaluation_curves.png",
        per_run_rows, summary_rows, timesteps, args.seeds,
        args.episodes_per_checkpoint)
    write_report(
        output_dir / "report.md", summary_rows, args.seeds, timesteps,
        args.episodes_per_checkpoint, args.evaluation_seed)
    print("Analyzed %d policies" % (
        len(args.seeds) * len(ALGORITHMS) * len(timesteps)))
    for name in (
            "greedy_evaluation_curves.png",
            "evaluation_curve_per_run.csv",
            "evaluation_curve_summary.csv",
            "evaluation_curve_paired_differences.csv", "report.md"):
        print(output_dir / name)


if __name__ == "__main__":
    main()
