#!/usr/bin/env python
"""Compare 256x256 and 128x128 networks using paired training seeds."""

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


ARCHITECTURES = ("large", "small")
ARCHITECTURE_LABELS = {
    "large": "256x256",
    "small": "128x128",
}
COLORS = {"large": "tab:gray", "small": "tab:purple"}


def load_architecture(root, prefix, architecture, seeds, timesteps,
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
                    "architecture": architecture,
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
            for architecture in ARCHITECTURES:
                values[architecture] = np.asarray([
                    row["evaluation_score_mean"] for row in rows
                    if row["algorithm"] == algorithm
                    and row["architecture"] == architecture
                    and row["training_timestep"] == timestep
                ], dtype=np.float64)
                if len(values[architecture]) != len(seeds):
                    raise ValueError(
                        "%s %s at %d has %d seeds; expected %d" % (
                            algorithm, architecture, timestep,
                            len(values[architecture]), len(seeds)))
                low, high = t_mean_ci(values[architecture])
                summaries.append({
                    "algorithm": algorithm,
                    "architecture": architecture,
                    "training_timestep": timestep,
                    "training_seeds": len(values[architecture]),
                    "mean_score": float(np.mean(values[architecture])),
                    "mean_ci95_low": low,
                    "mean_ci95_high": high,
                    "seed_mean_std": float(
                        np.std(values[architecture], ddof=1)),
                    "median_seed_mean": float(
                        np.median(values[architecture])),
                })
            difference = values["small"] - values["large"]
            low, high = t_mean_ci(difference)
            paired.append({
                "algorithm": algorithm,
                "training_timestep": timestep,
                "paired_training_seeds": len(seeds),
                "large_256_mean": float(np.mean(values["large"])),
                "small_128_mean": float(np.mean(values["small"])),
                "mean_difference_128_minus_256": float(
                    np.mean(difference)),
                "difference_ci95_low": low,
                "difference_ci95_high": high,
                "ci_excludes_zero": int(low > 0 or high < 0),
            })
    return summaries, paired


def plot(path, summaries, timesteps, seed_count):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for axis, algorithm in zip(axes, ALGORITHMS):
        for architecture in ARCHITECTURES:
            selected = sorted([
                row for row in summaries
                if row["algorithm"] == algorithm
                and row["architecture"] == architecture
            ], key=lambda row: row["training_timestep"])
            x = np.asarray([
                row["training_timestep"] for row in selected])
            mean = np.asarray([row["mean_score"] for row in selected])
            low = np.asarray([row["mean_ci95_low"] for row in selected])
            high = np.asarray([row["mean_ci95_high"] for row in selected])
            axis.fill_between(
                x, low, high, color=COLORS[architecture], alpha=0.14)
            axis.plot(
                x, mean, color=COLORS[architecture], linewidth=2.4,
                marker="o", label=ARCHITECTURE_LABELS[architecture])
        axis.set_title(LABELS[algorithm])
        axis.set_xlabel("Training timestep")
        axis.set_xticks(timesteps)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Mean greedy evaluation score")
    figure.suptitle(
        "Network-capacity pilot\nmean with 95% CI across %d paired seeds" %
        seed_count)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(path, paired):
    lines = [
        "# Network-capacity pilot\n\n",
        "Positive differences favor the 128x128 network. This is a "
        "three-seed pilot, so confidence intervals are descriptive and "
        "should not be treated as final evidence.\n\n",
        "| Algorithm | Timestep | 256x256 | 128x128 | Paired difference | 95% CI | Excludes zero |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in paired:
        lines.append(
            "| %s | %s | %.2f | %.2f | %.2f | %.2f–%.2f | %s |\n" % (
                LABELS[row["algorithm"]],
                format(row["training_timestep"], ","),
                row["large_256_mean"], row["small_128_mean"],
                row["mean_difference_128_minus_256"],
                row["difference_ci95_low"],
                row["difference_ci95_high"],
                "yes" if row["ci_excludes_zero"] else "no"))
    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 256x256 and 128x128 network pilots.")
    parser.add_argument("--root", default="state_experiments")
    parser.add_argument("--large-prefix", default="arch256")
    parser.add_argument("--small-prefix", default="arch128")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44])
    parser.add_argument("--timesteps", nargs="+", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=999)
    parser.add_argument("--episodes-per-checkpoint", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        default="state_experiments/architecture-pilot-500k")
    args = parser.parse_args()
    if args.timesteps != sorted(set(args.timesteps)):
        parser.error("--timesteps must be unique and strictly increasing")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if len(args.seeds) < 2:
        parser.error("at least two paired seeds are required")

    root = Path(args.root)
    rows = []
    rows.extend(load_architecture(
        root, args.large_prefix, "large", args.seeds,
        args.timesteps, args.evaluation_seed,
        args.episodes_per_checkpoint))
    rows.extend(load_architecture(
        root, args.small_prefix, "small", args.seeds,
        args.timesteps, args.evaluation_seed,
        args.episodes_per_checkpoint))
    summaries, paired = summarize(rows, args.seeds, args.timesteps)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "architecture_summary.csv", summaries)
    write_csv(output / "architecture_paired_differences.csv", paired)
    plot(
        output / "architecture_comparison.png", summaries,
        args.timesteps, len(args.seeds))
    write_report(output / "report.md", paired)
    print("Architecture pilot comparison complete:", output)


if __name__ == "__main__":
    main()

