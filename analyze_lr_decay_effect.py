#!/usr/bin/env python
"""Compare fixed-rate and decayed-rate long-horizon experiments."""

import argparse
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from analyze_evaluation_curves import (
    ALGORITHMS, LABELS, load_curve, t_mean_ci, write_csv,
)


PROTOCOLS = ("baseline", "lr_decay")
PROTOCOL_LABELS = {
    "baseline": "Fixed LR",
    "lr_decay": "LR decay",
}
COLORS = {"baseline": "tab:gray", "lr_decay": "tab:green"}


def load_protocol(root, prefix, protocol, seeds, timesteps,
                  evaluation_seed, episodes):
    rows = []
    for algorithm in ALGORITHMS:
        for seed in seeds:
            run_name = "%s-%s-seed%d" % (prefix, algorithm, seed)
            path = root / run_name / (
                "evaluation_curve_seed%d.csv" % evaluation_seed)
            points = load_curve(
                path, algorithm, run_name, evaluation_seed,
                timesteps, episodes)
            for point in points:
                point.update({
                    "protocol": protocol,
                    "algorithm": algorithm,
                    "seed": seed,
                })
                rows.append(point)
    return rows


def summarize(rows, seeds, timesteps):
    summaries = []
    paired = []
    for algorithm in ALGORITHMS:
        for timestep in timesteps:
            values = {}
            for protocol in PROTOCOLS:
                values[protocol] = np.asarray([
                    row["evaluation_score_mean"] for row in rows
                    if row["algorithm"] == algorithm
                    and row["protocol"] == protocol
                    and row["training_timestep"] == timestep
                ], dtype=np.float64)
                low, high = t_mean_ci(values[protocol])
                summaries.append({
                    "algorithm": algorithm,
                    "protocol": protocol,
                    "training_timestep": timestep,
                    "training_seeds": len(values[protocol]),
                    "mean_score": float(np.mean(values[protocol])),
                    "mean_ci95_low": low,
                    "mean_ci95_high": high,
                    "seed_mean_std": float(
                        np.std(values[protocol], ddof=1)),
                    "median_seed_mean": float(
                        np.median(values[protocol])),
                })
            difference = values["lr_decay"] - values["baseline"]
            low, high = t_mean_ci(difference)
            paired.append({
                "algorithm": algorithm,
                "training_timestep": timestep,
                "paired_training_seeds": len(seeds),
                "baseline_mean": float(np.mean(values["baseline"])),
                "lr_decay_mean": float(np.mean(values["lr_decay"])),
                "mean_difference_lr_decay_minus_baseline": float(
                    np.mean(difference)),
                "difference_ci95_low": low,
                "difference_ci95_high": high,
                "ci_excludes_zero": int(low > 0 or high < 0),
            })
    return summaries, paired


def plot(path, summaries, timesteps):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for axis, algorithm in zip(axes, ALGORITHMS):
        for protocol in PROTOCOLS:
            selected = sorted([
                row for row in summaries
                if row["algorithm"] == algorithm
                and row["protocol"] == protocol
            ], key=lambda row: row["training_timestep"])
            x = np.asarray([
                row["training_timestep"] for row in selected])
            mean = np.asarray([row["mean_score"] for row in selected])
            low = np.asarray([row["mean_ci95_low"] for row in selected])
            high = np.asarray([row["mean_ci95_high"] for row in selected])
            axis.fill_between(
                x, low, high, color=COLORS[protocol], alpha=0.14)
            axis.plot(
                x, mean, color=COLORS[protocol], linewidth=2.4,
                marker="o", label=PROTOCOL_LABELS[protocol])
        axis.set_title(LABELS[algorithm])
        axis.set_xlabel("Training timestep")
        axis.set_xticks(timesteps)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Mean greedy evaluation score")
    figure.suptitle(
        "Learning-rate decay ablation\nmean with 95% CI across paired seeds")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path, paired):
    lines = [
        "# Learning-rate decay ablation\n\n",
        "Positive differences favor learning-rate decay. The independent "
        "replicates are paired training seeds.\n\n",
        "| Algorithm | Timestep | Fixed LR | LR decay | Paired difference | 95% CI | Excludes zero |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in paired:
        lines.append(
            "| %s | %s | %.2f | %.2f | %.2f | %.2f–%.2f | %s |\n" % (
                LABELS[row["algorithm"]],
                format(row["training_timestep"], ","),
                row["baseline_mean"], row["lr_decay_mean"],
                row["mean_difference_lr_decay_minus_baseline"],
                row["difference_ci95_low"],
                row["difference_ci95_high"],
                "yes" if row["ci_excludes_zero"] else "no"))
    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Compare fixed and decayed learning-rate experiments.")
    parser.add_argument("--root", default="state_experiments")
    parser.add_argument("--baseline-prefix", default="long")
    parser.add_argument("--treatment-prefix", default="lrdecay")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--timesteps", nargs="+", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=999)
    parser.add_argument("--episodes-per-checkpoint", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="state_experiments/lrdecay-vs-fixed-1.5m")
    args = parser.parse_args()
    if args.timesteps != sorted(set(args.timesteps)):
        parser.error("--timesteps must be unique and strictly increasing")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")

    root = Path(args.root)
    rows = []
    rows.extend(load_protocol(
        root, args.baseline_prefix, "baseline", args.seeds,
        args.timesteps, args.evaluation_seed,
        args.episodes_per_checkpoint))
    rows.extend(load_protocol(
        root, args.treatment_prefix, "lr_decay", args.seeds,
        args.timesteps, args.evaluation_seed,
        args.episodes_per_checkpoint))
    summaries, paired = summarize(rows, args.seeds, args.timesteps)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "lr_decay_summary.csv", summaries)
    write_csv(output / "lr_decay_paired_differences.csv", paired)
    plot(output / "lr_decay_comparison.png", summaries, args.timesteps)
    write_report(output / "report.md", paired)
    print("Learning-rate decay comparison complete:", output)


if __name__ == "__main__":
    main()
