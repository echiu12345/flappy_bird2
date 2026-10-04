#!/usr/bin/env python
"""Create fixed-budget, multi-seed DQN versus Double DQN reports."""

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("dqn", "ddqn")
LABELS = {"dqn": "DQN", "ddqn": "Double DQN"}
COLORS = {"dqn": "tab:blue", "ddqn": "tab:orange"}
METRICS = ("score", "mean_max_q", "mean_loss")


def parse_number(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def parse_integer(value, field, path):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s has an invalid %s value: %r" % (
            path, field, value))
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError("%s has a non-integer %s value: %r" % (
            path, field, value))
    return int(parsed)


def parse_boolean(value, field, path):
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y"):
        return True
    if normalized in ("0", "false", "no", "n", ""):
        return False
    raise ValueError("%s has an invalid %s value: %r" % (
        path, field, value))


def require_strictly_increasing(values, field, path):
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError(
            "%s must contain unique, strictly increasing %s values" % (
                path, field))


def load_run(path, algorithm, seed, max_timestep):
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("No completed episodes found in " + str(path))

    required = {
        "algorithm", "representation", "seed", "timestep", "episode", "score",
        "mean_max_q", "mean_loss", "episode_frames",
    }
    missing_columns = required.difference(reader.fieldnames or ())
    if missing_columns:
        raise ValueError(
            "%s is missing columns: %s" % (
                path, ", ".join(sorted(missing_columns))))

    recorded_algorithms = {
        str(row["algorithm"]).strip().lower() for row in rows
    }
    recorded_seeds = {
        parse_integer(row["seed"], "seed", path) for row in rows
    }
    representations = {
        str(row["representation"]).strip().lower() for row in rows
    }
    if len(recorded_algorithms) != 1 or len(recorded_seeds) != 1:
        raise ValueError("Mixed algorithms or seeds in " + str(path))
    if len(representations) != 1 or "" in representations:
        raise ValueError("Mixed or empty representations in " + str(path))
    recorded_algorithm = next(iter(recorded_algorithms))
    recorded_seed = next(iter(recorded_seeds))
    if recorded_algorithm != algorithm or recorded_seed != seed:
        raise ValueError(
            "%s records %s seed %d, expected %s seed %d" % (
                path, recorded_algorithm, recorded_seed, algorithm, seed))

    episodes = [
        parse_integer(row["episode"], "episode", path) for row in rows
    ]
    timesteps = [
        parse_integer(row["timestep"], "timestep", path) for row in rows
    ]
    frames = [parse_number(row["episode_frames"]) for row in rows]
    if episodes[0] <= 0 or timesteps[0] <= 0:
        raise ValueError("%s contains non-positive episode or timestep values" % path)
    if any(not math.isfinite(value) or value <= 0 for value in frames):
        raise ValueError("%s contains non-positive episode_frames" % path)
    scores = np.asarray(
        [parse_number(row["score"]) for row in rows], dtype=np.float64)
    if np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("%s contains invalid training scores" % path)
    require_strictly_increasing(episodes, "episode", path)
    require_strictly_increasing(timesteps, "timestep", path)

    truncated = [False] * len(rows)
    if "truncated" in (reader.fieldnames or ()):
        truncated = [
            parse_boolean(row["truncated"], "truncated", path)
            for row in rows
        ]
        truncated_indices = [
            index for index, value in enumerate(truncated) if value
        ]
        if truncated_indices and truncated_indices != [len(rows) - 1]:
            raise ValueError(
                "%s may mark only its final row as truncated" % path)

    observed_until = timesteps[-1]
    if observed_until < max_timestep:
        raise ValueError(
            "%s is incomplete: %s of %s timesteps" % (
                path, format(observed_until, ","),
                format(max_timestep, ",")))

    selected_indices = [
        index for index, timestep in enumerate(timesteps)
        if timestep <= max_timestep
    ]
    if not selected_indices:
        raise ValueError("No episodes are inside the selected range for " + str(path))
    selected = [rows[index] for index in selected_indices]

    data = {
        "algorithm": algorithm,
        "seed": seed,
        "path": str(path),
        "observed_until": observed_until,
        "representation": next(iter(representations)),
        "timestep": np.asarray(
            [timesteps[index] for index in selected_indices], dtype=np.float64),
        "episode": np.asarray(
            [episodes[index] for index in selected_indices], dtype=np.float64),
    }
    for metric in METRICS:
        if metric == "score":
            data[metric] = scores[selected_indices]
        else:
            data[metric] = np.asarray(
                [parse_number(row[metric]) for row in selected],
                dtype=np.float64)
    data["episode_frames"] = np.asarray(
        [frames[index] for index in selected_indices],
        dtype=np.float64)
    data["score_valid"] = np.asarray(
        [not truncated[index] for index in selected_indices], dtype=bool)
    return data


def load_evaluation(path, algorithm, training_seed, evaluation_seed,
                    expected_run_name, expected_episodes=None):
    if not path.is_file():
        return None
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("No evaluation episodes found in " + str(path))
    required = {
        "algorithm", "run_name", "evaluation_seed", "episode", "score",
        "frames",
    }
    missing_columns = required.difference(reader.fieldnames or ())
    if missing_columns:
        raise ValueError("%s is missing columns: %s" % (
            path, ", ".join(sorted(missing_columns))))
    if any(row["algorithm"].lower() != algorithm for row in rows):
        raise ValueError("Mixed or incorrect algorithms in " + str(path))
    if any(row["run_name"] != expected_run_name for row in rows):
        raise ValueError("Mixed or incorrect run_name values in " + str(path))
    if any(parse_integer(
            row["evaluation_seed"], "evaluation_seed", path
            ) != evaluation_seed for row in rows):
        raise ValueError("Incorrect evaluation seed in " + str(path))
    for optional_seed_column in ("seed", "training_seed"):
        if optional_seed_column in (reader.fieldnames or ()):
            if any(parse_integer(
                    row[optional_seed_column], optional_seed_column, path
                    ) != training_seed for row in rows):
                raise ValueError(
                    "Incorrect %s in %s" % (optional_seed_column, path))

    episodes = [
        parse_integer(row["episode"], "episode", path) for row in rows
    ]
    require_strictly_increasing(episodes, "episode", path)
    if episodes[0] != 1 or episodes[-1] != len(episodes):
        raise ValueError(
            "%s evaluation episodes must be consecutively numbered from 1" %
            path)
    if expected_episodes is not None and len(rows) != expected_episodes:
        raise ValueError(
            "%s contains %d evaluation episodes; expected %d" % (
                path, len(rows), expected_episodes))
    frames = [parse_number(row["frames"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0 for value in frames):
        raise ValueError("%s contains non-positive evaluation frames" % path)
    scores = np.asarray(
        [parse_number(row["score"]) for row in rows], dtype=np.float64)
    if np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("%s contains invalid evaluation scores" % path)
    return scores


def trailing_timestep_curves(run, grid, window_steps):
    curves = {
        metric: np.full(len(grid), np.nan, dtype=np.float64)
        for metric in METRICS
    }
    for index, endpoint in enumerate(grid):
        selected = (
            (run["timestep"] > endpoint - window_steps)
            & (run["timestep"] <= endpoint)
        )
        if not np.any(selected):
            continue
        score_selected = selected & run["score_valid"]
        score_values = run["score"][score_selected]
        score_values = score_values[np.isfinite(score_values)]
        if len(score_values):
            curves["score"][index] = float(np.mean(score_values))
        for metric in ("mean_max_q", "mean_loss"):
            values = run[metric][selected]
            weights = run["episode_frames"][selected]
            finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            if np.any(finite):
                curves[metric][index] = np.average(
                    values[finite], weights=weights[finite])
    return curves


def first_sustained_threshold(grid, values, threshold, consecutive):
    reached = np.asarray(
        np.isfinite(values) & (values >= threshold), dtype=np.int8)
    if len(reached) < consecutive:
        return None
    streaks = np.convolve(
        reached, np.ones(consecutive, dtype=np.int8), mode="valid")
    matches = np.flatnonzero(streaks == consecutive)
    if len(matches) == 0:
        return None
    return int(grid[int(matches[0])])


def normalized_auc(grid, values):
    """Return AUC/span, interpolating only gaps between finite endpoints."""
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2 or not finite[0] or not finite[-1]:
        return float("nan")
    filled = np.interp(grid, grid[finite], values[finite])
    span = float(grid[-1] - grid[0])
    if span <= 0:
        return float("nan")
    return float(np.trapz(filled, grid) / span)


def calculate_run_metrics(run, grid, window_steps, thresholds):
    curves = trailing_timestep_curves(run, grid, window_steps)
    score_curve = curves["score"]
    late_mask = grid >= 0.8 * grid[-1]
    late_values = score_curve[late_mask]
    finite_peak_indices = np.flatnonzero(np.isfinite(score_curve))
    if len(finite_peak_indices):
        peak_index = int(finite_peak_indices[
            np.argmax(score_curve[finite_peak_indices])])
        peak_score = float(score_curve[peak_index])
        peak_timestep = int(grid[peak_index])
    else:
        peak_score = float("nan")
        peak_timestep = None
    finite_late = late_values[np.isfinite(late_values)]
    late_mean = (
        float(np.mean(finite_late)) if len(finite_late) else float("nan"))
    late_episode_mask = (
        (run["timestep"] > 0.8 * grid[-1])
        & (run["timestep"] <= grid[-1])
        & run["score_valid"]
        & np.isfinite(run["score"])
    )
    late_episode_scores = run["score"][late_episode_mask]
    late_episode_std = (
        float(np.std(late_episode_scores, ddof=0))
        if len(late_episode_scores) else float("nan"))
    degradation = (
        max(0.0, peak_score - late_mean) / peak_score * 100.0
        if math.isfinite(peak_score) and math.isfinite(late_mean)
        and peak_score > 0 else (
            0.0 if peak_score == 0 and math.isfinite(late_mean)
            else float("nan")))
    average_training_score = normalized_auc(grid, score_curve)

    metrics = {
        "algorithm": run["algorithm"],
        "seed": run["seed"],
        "episodes": int(np.count_nonzero(run["score_valid"])),
        "truncated_rows_excluded": int(np.count_nonzero(~run["score_valid"])),
        "last_logged_timestep": int(run["observed_until"]),
        "last_in_budget_timestep": int(run["timestep"][-1]),
        "average_training_score": average_training_score,
        "final_rolling_score": float(score_curve[-1]),
        "late_score_mean": late_mean,
        "late_episode_score_std": late_episode_std,
        "peak_rolling_score": peak_score,
        "peak_timestep": peak_timestep,
        "peak_to_late_drop_percent": degradation,
        "final_mean_max_q": float(curves["mean_max_q"][-1]),
        "final_mean_loss": float(curves["mean_loss"][-1]),
    }
    evaluation_scores = run.get("evaluation_scores")
    if evaluation_scores is None:
        metrics.update({
            "evaluation_episodes": 0,
            "evaluation_score_mean": float("nan"),
            "evaluation_score_median": float("nan"),
            "evaluation_score_std": float("nan"),
            "evaluation_score_max": float("nan"),
        })
    else:
        metrics.update({
            "evaluation_episodes": len(evaluation_scores),
            "evaluation_score_mean": float(np.mean(evaluation_scores)),
            "evaluation_score_median": float(np.median(evaluation_scores)),
            "evaluation_score_std": float(np.std(evaluation_scores, ddof=0)),
            "evaluation_score_max": float(np.max(evaluation_scores)),
        })
    for threshold in thresholds:
        key = "time_to_score_" + str(threshold).replace(".", "_")
        metrics[key] = first_sustained_threshold(
            grid, score_curve, threshold, consecutive=3)
    return curves, metrics


def finite_mean_std(values):
    finite = np.asarray(
        [value for value in values if value is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return mean, std


def column_mean_std(matrix):
    """Compute finite-only column statistics without all-NaN warnings."""
    means = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    stds = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for index in range(matrix.shape[1]):
        values = matrix[:, index]
        values = values[np.isfinite(values)]
        if len(values):
            means[index] = float(np.mean(values))
            stds[index] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    return means, stds


def aggregate_metrics(per_run_metrics, thresholds):
    metric_names = [
        "average_training_score", "final_rolling_score",
        "late_score_mean", "late_episode_score_std", "peak_rolling_score",
        "peak_timestep", "peak_to_late_drop_percent",
        "final_mean_max_q", "final_mean_loss",
        "evaluation_score_mean", "evaluation_score_median",
        "evaluation_score_std", "evaluation_score_max",
    ]
    metric_names.extend(
        "time_to_score_" + str(value).replace(".", "_")
        for value in thresholds)
    summary = []
    for algorithm in ALGORITHMS:
        rows = [
            row for row in per_run_metrics
            if row["algorithm"] == algorithm
        ]
        if not rows:
            continue
        result = {"algorithm": algorithm, "runs": len(rows)}
        for metric in metric_names:
            mean, std = finite_mean_std([row[metric] for row in rows])
            result[metric + "_mean"] = mean
            result[metric + "_std"] = std
        summary.append(result)
    return summary


def format_number(value, decimals=2):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return format(float(value), ",." + str(decimals) + "f")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(path, runs, run_curves, grid, window_steps,
                         max_timestep):
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    plot_specs = (
        ("score", "Score per episode", "Training performance"),
        ("mean_max_q", "Mean predicted max Q", "Value estimates"),
        ("mean_loss", "Huber loss", "Optimization behavior"),
    )

    for axis, (metric, ylabel, title) in zip(axes, plot_specs):
        for algorithm in ALGORITHMS:
            selected = [
                (run, curves)
                for run, curves in zip(runs, run_curves)
                if run["algorithm"] == algorithm
            ]
            if not selected:
                continue
            matrix = np.vstack([curves[metric] for _, curves in selected])
            for _, curves in selected:
                axis.plot(
                    grid, curves[metric], color=COLORS[algorithm],
                    alpha=0.16, linewidth=0.9)
            mean, std = column_mean_std(matrix)
            axis.fill_between(
                grid, mean - std, mean + std,
                color=COLORS[algorithm], alpha=0.16)
            axis.plot(
                grid, mean, color=COLORS[algorithm], linewidth=2.3,
                label="%s mean ± 1 SD (n=%d)" % (
                    LABELS[algorithm], len(selected)))
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[0].axhline(10, color="gray", linestyle="--", linewidth=1,
                    alpha=0.7, label="score criterion = 10")
    axes[0].legend()
    axes[-1].set_xlabel("Environment timestep")
    axes[-1].set_xlim(0, max_timestep)
    figure.suptitle(
        "Flappy Bird DQN vs Double DQN — fixed %s-step budget\n"
        "%s-step trailing windows, faint lines are individual seeds" % (
            format(max_timestep, ","), format(window_steps, ",")),
        fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_summary(path, per_run_metrics):
    specifications = (
        ("average_training_score", "Average training score\n(area under learning curve)", True),
        ("late_score_mean", "Mean score in final 20%", True),
        ("late_episode_score_std", "Late per-episode score SD\n(final 20%)", False),
        ("peak_to_late_drop_percent", "Drop from peak to late mean (%)", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    rng = np.random.default_rng(7)

    for axis, (metric, title, higher_is_better) in zip(
            axes.flat, specifications):
        algorithms = [
            algorithm for algorithm in ALGORITHMS
            if any(
                row["algorithm"] == algorithm
                and math.isfinite(float(row[metric]))
                for row in per_run_metrics)
        ]
        means = []
        errors = []
        for algorithm in algorithms:
            values = np.asarray([
                row[metric] for row in per_run_metrics
                if row["algorithm"] == algorithm
            ], dtype=np.float64)
            mean, std = finite_mean_std(values)
            means.append(mean)
            errors.append(std)
        positions = np.arange(len(algorithms))
        axis.bar(
            positions, means, yerr=errors,
            color=[COLORS[value] for value in algorithms],
            alpha=0.72, capsize=5)
        for position, algorithm in zip(positions, algorithms):
            values = [
                row[metric] for row in per_run_metrics
                if row["algorithm"] == algorithm
                and math.isfinite(float(row[metric]))
            ]
            jitter = rng.uniform(-0.06, 0.06, size=len(values))
            axis.scatter(
                position + jitter, values, color="black", s=26,
                alpha=0.72, zorder=3)
        axis.set_xticks(positions, [LABELS[value] for value in algorithms])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_xlabel(direction)

    figure.suptitle(
        "Training efficiency and stability summary\n"
        "bars show seed mean ± 1 SD; dots show individual seeds",
        fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_evaluation(path, per_run_metrics):
    usable = [
        row for row in per_run_metrics
        if math.isfinite(row["evaluation_score_mean"])
    ]
    if not usable:
        return False

    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    positions = {"dqn": 0, "ddqn": 1}
    paired_seeds = sorted(set(
        row["seed"] for row in usable
        if all(any(
            candidate["seed"] == row["seed"]
            and candidate["algorithm"] == algorithm
            for candidate in usable) for algorithm in ALGORITHMS)
    ))
    for seed in paired_seeds:
        values = {
            row["algorithm"]: row["evaluation_score_mean"]
            for row in usable if row["seed"] == seed
        }
        axis.plot(
            [positions["dqn"], positions["ddqn"]],
            [values["dqn"], values["ddqn"]],
            color="gray", alpha=0.45, linewidth=1.2)
        for algorithm in ALGORITHMS:
            axis.scatter(
                positions[algorithm], values[algorithm],
                color=COLORS[algorithm], s=55, zorder=3)
            axis.annotate(
                "seed %d" % seed,
                (positions[algorithm], values[algorithm]),
                xytext=(5, 3), textcoords="offset points", fontsize=8,
                alpha=0.75)

    for algorithm in ALGORITHMS:
        values = [
            row["evaluation_score_mean"] for row in usable
            if row["algorithm"] == algorithm
        ]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        axis.errorbar(
            positions[algorithm], mean, yerr=std, fmt="_", markersize=28,
            markeredgewidth=3, capsize=7, color="black", zorder=4)

    axis.set_xticks([0, 1], [LABELS["dqn"], LABELS["ddqn"]])
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylabel("Mean greedy evaluation score per training seed")
    axis.set_title(
        "Held-out greedy policy evaluation\n"
        "lines pair identical training seeds; black marks show mean ± 1 SD")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def write_markdown_report(path, summary, seeds, max_timestep,
                          window_steps, grid_step, thresholds):
    by_algorithm = {row["algorithm"]: row for row in summary}
    lines = [
        "# DQN vs Double DQN comparison\n",
        "\n",
        "## Experimental protocol\n",
        "\n",
        "- Environment budget: **%s timesteps per run**.\n" %
        format(max_timestep, ","),
        "- Seeds: **%s**.\n" % ", ".join(str(seed) for seed in seeds),
        "- Smoothing: episodes completed in the trailing **%s timesteps**.\n" %
        format(window_steps, ","),
        "- Cross-seed alignment: one trailing-window estimate every **%s timesteps**.\n" %
        format(grid_step, ","),
        "- Uncertainty in figures: **±1 sample standard deviation**.\n",
        "\n",
        "## Aggregate results\n",
        "\n",
        "| Algorithm | Runs | Greedy eval. score | Avg. training score | Late score | Late per-episode SD | Peak-to-late drop |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for algorithm in ALGORITHMS:
        row = by_algorithm.get(algorithm)
        if not row:
            continue
        lines.append(
            "| %s | %d | %s ± %s | %s ± %s | %s ± %s | %s ± %s | %s%% ± %s%% |\n" % (
                LABELS[algorithm], row["runs"],
                format_number(row["evaluation_score_mean_mean"]),
                format_number(row["evaluation_score_mean_std"]),
                format_number(row["average_training_score_mean"]),
                format_number(row["average_training_score_std"]),
                format_number(row["late_score_mean_mean"]),
                format_number(row["late_score_mean_std"]),
                format_number(row["late_episode_score_std_mean"]),
                format_number(row["late_episode_score_std_std"]),
                format_number(row["peak_to_late_drop_percent_mean"]),
                format_number(row["peak_to_late_drop_percent_std"])))
    lines.extend([
        "\n",
        "## Metric definitions\n",
        "\n",
        "- **Average training score** is the timestep-normalized area under the fixed-window score curve; it measures sample efficiency after the initial window.\n",
        "- **Late score** is the mean smoothed score during the final 20% of the timestep budget.\n",
        "- **Late per-episode SD** is the standard deviation of complete episode scores ending during the final 20% of the timestep budget; lower values indicate less episode-to-episode variation.\n",
        "- **Peak-to-late drop** measures degradation after the best observed smoothed score.\n",
        "- **Greedy evaluation score** is computed with epsilon 0 on held-out episodes; each plotted dot is one independently trained seed.\n",
        "- **Time to criterion** is the first of three consecutive %s-step grid points at or above a score threshold (%s); see `per_run_metrics.csv`.\n" % (
            format(grid_step, ","),
            ", ".join(str(value) for value in thresholds)),
        "\n",
        "Predicted Q values are diagnostic estimates. A higher Q curve without a corresponding score improvement is consistent with overestimation, but it does not by itself prove overestimation because the true action values are unknown.\n",
    ])
    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze fixed-budget, multi-seed state DQN experiments.")
    parser.add_argument("--root", default="state_experiments")
    parser.add_argument(
        "--run-prefix", default="state",
        help="run-name prefix before -dqn-seedN and -ddqn-seedN")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-timestep", type=int, default=500000)
    parser.add_argument("--window-steps", type=int, default=50000)
    parser.add_argument("--grid-step", type=int, default=10000)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[5, 10])
    parser.add_argument("--evaluation-seed", type=int, default=999)
    parser.add_argument(
        "--expected-evaluation-episodes", type=int,
        help=("Require this many rows in every evaluation CSV. When omitted, "
              "all available evaluation CSVs must still have equal lengths."))
    parser.add_argument(
        "--output-dir", default="state_experiments/comparison-500k")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="Generate a progress report from completed files that exist")
    args = parser.parse_args()

    if args.max_timestep <= 0 or args.grid_step <= 0:
        parser.error("--max-timestep and --grid-step must be positive")
    if args.window_steps <= 0 or args.window_steps >= args.max_timestep:
        parser.error("--window-steps must be positive and below --max-timestep")
    if (args.expected_evaluation_episodes is not None
            and args.expected_evaluation_episodes <= 0):
        parser.error("--expected-evaluation-episodes must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    root = Path(args.root)
    runs = []
    missing = []
    for algorithm in ALGORITHMS:
        for seed in args.seeds:
            path = root / (
                "%s-%s-seed%d" % (args.run_prefix, algorithm, seed)
            ) / "episodes.csv"
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                run = load_run(path, algorithm, seed, args.max_timestep)
                evaluation_path = path.parent / (
                    "evaluation_seed%d.csv" % args.evaluation_seed)
                run["evaluation_scores"] = load_evaluation(
                    evaluation_path, algorithm, seed, args.evaluation_seed,
                    path.parent.name, args.expected_evaluation_episodes)
                runs.append(run)
            except ValueError as error:
                if args.allow_missing:
                    print("Skipping:", error)
                else:
                    raise

    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "Missing experiment logs:\n" + "\n".join(missing))
    if not runs:
        raise ValueError("No complete experiment runs were found")

    representations = {run["representation"] for run in runs}
    if len(representations) != 1:
        raise ValueError(
            "Runs use different state representations: %s" %
            ", ".join(sorted(representations)))
    evaluation_lengths = {
        len(run["evaluation_scores"])
        for run in runs if run.get("evaluation_scores") is not None
    }
    if len(evaluation_lengths) > 1:
        raise ValueError(
            "Evaluation CSVs have unequal episode counts: %s" %
            ", ".join(str(value) for value in sorted(evaluation_lengths)))

    grid = np.arange(
        args.window_steps, args.max_timestep + args.grid_step, args.grid_step,
        dtype=np.float64)
    if grid[-1] > args.max_timestep:
        grid[-1] = args.max_timestep
    grid = np.unique(grid)

    run_curves = []
    per_run_metrics = []
    for run in runs:
        curves, metrics = calculate_run_metrics(
            run, grid, args.window_steps, args.thresholds)
        run_curves.append(curves)
        per_run_metrics.append(metrics)
    summary = aggregate_metrics(per_run_metrics, args.thresholds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_run_metrics.csv", per_run_metrics)
    write_csv(output_dir / "algorithm_summary.csv", summary)
    plot_learning_curves(
        output_dir / "learning_curves.png", runs, run_curves, grid,
        args.window_steps, args.max_timestep)
    plot_summary(output_dir / "summary_metrics.png", per_run_metrics)
    has_evaluation_plot = plot_evaluation(
        output_dir / "evaluation_scores.png", per_run_metrics)
    write_markdown_report(
        output_dir / "report.md", summary, args.seeds,
        args.max_timestep, args.window_steps, args.grid_step,
        args.thresholds)

    print("Analyzed %d runs" % len(runs))
    filenames = [
        "learning_curves.png", "summary_metrics.png",
        "per_run_metrics.csv", "algorithm_summary.csv", "report.md",
    ]
    if has_evaluation_plot:
        filenames.insert(2, "evaluation_scores.png")
    for filename in filenames:
        print(output_dir / filename)


if __name__ == "__main__":
    main()
