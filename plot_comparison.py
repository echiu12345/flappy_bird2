#!/usr/bin/env python
import argparse
import csv
import os

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def read_log(path, max_timestep=None):
    with open(path, "r", newline="") as log_file:
        rows = list(csv.DictReader(log_file))
    if max_timestep is not None:
        rows = [
            row for row in rows
            if float(row["timestep"]) <= max_timestep
        ]
    if not rows:
        raise ValueError(
            "No completed episodes found in the selected range for " + path)
    return {
        "timestep": np.asarray([float(row["timestep"]) for row in rows]),
        "score": np.asarray([float(row["score"]) for row in rows]),
        "mean_max_q": np.asarray(
            [float(row["mean_max_q"]) for row in rows]),
    }


def rolling_mean(values, window):
    if len(values) < window:
        return np.asarray([])
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_metric(axis, data, metric, label, color, window):
    axis.plot(
        data["timestep"], data[metric], color=color, alpha=0.18,
        linewidth=0.8)
    smoothed = rolling_mean(data[metric], window)
    if len(smoothed):
        smoothed_timesteps = data["timestep"][window - 1:]
        axis.plot(
            smoothed_timesteps, smoothed, color=color,
            linewidth=2, label=label + " rolling mean")
        if metric == "score":
            peak_index = int(np.argmax(smoothed))
            peak_x = smoothed_timesteps[peak_index]
            peak_y = smoothed[peak_index]
            axis.scatter([peak_x], [peak_y], color=color, s=36, zorder=4)
            axis.annotate(
                "%s peak %.2f\n@ %s steps" % (
                    label, peak_y, format(int(peak_x), ",")),
                xy=(peak_x, peak_y), xytext=(8, 10),
                textcoords="offset points", color=color, fontsize=9)
    else:
        axis.plot([], [], color=color, linewidth=2, label=label)


def main():
    parser = argparse.ArgumentParser(
        description="Plot DQN/DDQN training progress and comparisons.")
    parser.add_argument("--dqn", help="DQN episodes.csv path")
    parser.add_argument("--ddqn", help="DDQN episodes.csv path")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--max-timestep", type=int,
        help="Only plot completed episodes at or before this timestep")
    parser.add_argument(
        "--output", default=os.path.join("experiments", "dqn_vs_ddqn.png"))
    args = parser.parse_args()

    if not args.dqn and not args.ddqn:
        parser.error("provide --dqn, --ddqn, or both")
    if args.window < 1:
        parser.error("--window must be at least 1")

    datasets = []
    if args.dqn:
        datasets.append((
            read_log(args.dqn, args.max_timestep), "DQN", "tab:blue"))
    if args.ddqn:
        datasets.append((
            read_log(args.ddqn, args.max_timestep),
            "Double DQN", "tab:orange"))
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for data, label, color in datasets:
        plot_metric(axes[0], data, "score", label, color, args.window)
    axes[0].set_ylabel("Score per episode")
    axes[0].set_title(
        "Flappy Bird: DQN vs Double DQN"
        if len(datasets) == 2 else "Flappy Bird: " + datasets[0][1] + " progress")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for data, label, color in datasets:
        plot_metric(
            axes[1], data, "mean_max_q", label, color, args.window)
    axes[1].set_xlabel("Environment timestep")
    axes[1].set_ylabel("Mean predicted max Q")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    if args.max_timestep is not None:
        axes[1].set_xlim(0, args.max_timestep)

    output_directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_directory, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    print("Saved comparison graph to", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
