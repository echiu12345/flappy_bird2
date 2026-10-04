#!/usr/bin/env python
"""Train and evaluate DQN/Double DQN agents from four numeric state features."""

import argparse
import csv
import json
import os
import random
import sys
from collections import deque
from itertools import cycle

import numpy as np
import tensorflow

tf = tensorflow.compat.v1
tf.disable_v2_behavior()

STATE_DIM = 4
ACTION_DIM = 2
STATE_EXPERIMENT_DIR = "state_experiments"
RUN_CONFIG_FILENAME = "run_config.json"
TRAIN_LOG_FIELDS = [
    "algorithm", "representation", "seed", "timestep", "episode",
    "score", "episode_reward", "episode_frames", "epsilon", "learning_rate",
    "mean_max_q", "mean_loss", "truncated",
]
EVALUATION_FIELDS = [
    "algorithm", "run_name", "evaluation_seed", "eval_episode_seed",
    "episode", "score", "frames", "truncated",
]
STEP_EVALUATION_FIELDS = [
    "algorithm", "run_name", "training_timestep", "evaluation_seed",
    "eval_episode_seed", "episode", "score", "frames", "truncated",
]


def create_network(scope, hidden_size, trainable):
    with tf.variable_scope(scope):
        state = tf.placeholder(tf.float32, [None, STATE_DIM], name="state")
        initializer = tf.glorot_uniform_initializer()

        w1 = tf.get_variable(
            "w1", [STATE_DIM, hidden_size], initializer=initializer,
            trainable=trainable)
        b1 = tf.get_variable(
            "b1", [hidden_size], initializer=tf.zeros_initializer(),
            trainable=trainable)
        hidden1 = tf.nn.relu(tf.matmul(state, w1) + b1)

        w2 = tf.get_variable(
            "w2", [hidden_size, hidden_size], initializer=initializer,
            trainable=trainable)
        b2 = tf.get_variable(
            "b2", [hidden_size], initializer=tf.zeros_initializer(),
            trainable=trainable)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)

        w3 = tf.get_variable(
            "w3", [hidden_size, ACTION_DIM], initializer=initializer,
            trainable=trainable)
        b3 = tf.get_variable(
            "b3", [ACTION_DIM], initializer=tf.zeros_initializer(),
            trainable=trainable)
        q_values = tf.matmul(hidden2, w3) + b3

    return state, q_values, [w1, b1, w2, b2, w3, b3]


def create_training_graph(args):
    online_state, online_q, online_parameters = create_network(
        "online", args.hidden_size, trainable=True)
    target_state, target_q, target_parameters = create_network(
        "target", args.hidden_size, trainable=False)

    actions = tf.placeholder(tf.int32, [None], name="actions")
    targets = tf.placeholder(tf.float32, [None], name="targets")
    action_mask = tf.one_hot(actions, ACTION_DIM, dtype=tf.float32)
    selected_q = tf.reduce_sum(online_q * action_mask, axis=1)
    loss = tf.losses.huber_loss(targets, selected_q)

    learning_rate = tf.placeholder_with_default(
        tf.constant(args.learning_rate, dtype=tf.float32), shape=(),
        name="learning_rate")
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss, var_list=online_parameters)
    clipped_gradients = [
        (tf.clip_by_norm(gradient, 1.0), variable)
        for gradient, variable in gradients
        if gradient is not None
    ]
    train_step = optimizer.apply_gradients(clipped_gradients)

    hard_update = tf.group(*[
        target.assign(online)
        for online, target in zip(online_parameters, target_parameters)
    ])
    with tf.control_dependencies([train_step]):
        soft_update = tf.group(*[
            target.assign(args.tau * online + (1.0 - args.tau) * target)
            for online, target in zip(online_parameters, target_parameters)
        ])

    return {
        "online_state": online_state,
        "online_q": online_q,
        "target_state": target_state,
        "target_q": target_q,
        "actions": actions,
        "targets": targets,
        "learning_rate": learning_rate,
        "loss": loss,
        "train_step": train_step,
        "hard_update": hard_update,
        "soft_update": soft_update,
    }


def calculate_targets(algorithm, rewards, terminals, online_q, target_q, gamma):
    rewards = np.asarray(rewards, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.float32)
    if algorithm == "ddqn":
        best_actions = np.argmax(online_q, axis=1)
        next_values = target_q[np.arange(len(best_actions)), best_actions]
    else:
        next_values = np.max(target_q, axis=1)
    return (rewards + (1.0 - terminals) * gamma * next_values).astype(
        np.float32)


def run_paths(run_name):
    run_directory = os.path.join(STATE_EXPERIMENT_DIR, run_name)
    return (
        run_directory,
        os.path.join(run_directory, "checkpoints"),
        os.path.join(run_directory, "episodes.csv"),
    )


def scientific_run_config(args):
    """Return immutable settings that define a comparable training run."""
    return {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "environment": "game.wrapped_flappy_bird",
        "representation": "state4",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "seed": args.seed,
        "budget": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
        },
        "network": {
            "hidden_size": args.hidden_size,
        },
        "replay": {
            "batch_size": args.batch_size,
            "memory_size": args.memory_size,
            "learning_starts": args.learning_starts,
            "train_frequency": args.train_frequency,
        },
        "optimization": {
            "optimizer": "adam",
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": [
                {"timestep": timestep, "learning_rate": learning_rate}
                for timestep, learning_rate
                in args.learning_rate_schedule_points
            ],
            "loss": "huber",
            "gradient_clip_norm": 1.0,
        },
        "returns": {
            "gamma": args.gamma,
        },
        "target_network": {
            "update": "soft_after_gradient_step",
            "tau": args.tau,
        },
        "artifacts": {
            "step_checkpoint_interval": args.step_checkpoint_interval,
        },
        "exploration": {
            "schedule": "linear_by_environment_timestep_after_warmup",
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay_steps": args.epsilon_decay_steps,
        },
        "reward": {
            "alive": 0.1,
            "pipe": 1.0,
            "terminal": -1.0,
        },
    }


def ensure_run_config(args, run_directory, has_existing_artifacts):
    """Persist a new run config and prevent unsafe mismatched resumes."""
    config_path = os.path.join(run_directory, RUN_CONFIG_FILENAME)
    requested_config = scientific_run_config(args)

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as config_file:
            saved_config = json.load(config_file)
        if saved_config != requested_config:
            raise ValueError(
                "Run configuration does not match the saved run_config.json. "
                "Use the original arguments or choose a new --run-name.\n"
                "Saved configuration:\n%s\nRequested configuration:\n%s" % (
                    json.dumps(saved_config, indent=2, sort_keys=True),
                    json.dumps(requested_config, indent=2, sort_keys=True)))
        return config_path

    if has_existing_artifacts:
        raise ValueError(
            "This legacy run has checkpoints or a CSV but no run_config.json, "
            "so its scientific settings cannot be verified for a safe resume. "
            "Choose a new --run-name for a controlled run.")

    temporary_path = config_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as config_file:
        json.dump(requested_config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")
    os.replace(temporary_path, config_path)
    return config_path


def load_log_position(log_path):
    if not os.path.isfile(log_path):
        return 0, 0
    with open(log_path, "r", newline="") as log_file:
        rows = list(csv.DictReader(log_file))
    if not rows:
        return 0, 0
    return int(rows[-1]["episode"]), int(rows[-1]["timestep"])


def select_action(session, graph, state, epsilon):
    q_values = session.run(
        graph["online_q"],
        feed_dict={graph["online_state"]: [state]})[0]
    if random.random() < epsilon:
        return random.randrange(ACTION_DIM), q_values
    return int(np.argmax(q_values)), q_values


def epsilon_at_timestep(args, timestep):
    """Return a shared linear exploration schedule based on interactions."""
    decay_timestep = max(0, timestep - args.learning_starts)
    progress = min(
        1.0, decay_timestep / float(args.epsilon_decay_steps))
    return args.epsilon_start + progress * (
        args.epsilon_end - args.epsilon_start)


def learning_rate_at_timestep(args, timestep):
    """Return the piecewise-constant learning rate for a global timestep."""
    learning_rate = args.learning_rate
    for milestone, scheduled_rate in args.learning_rate_schedule_points:
        if timestep < milestone:
            break
        learning_rate = scheduled_rate
    return learning_rate


def parse_learning_rate_schedule(values, initial_rate, max_steps):
    """Parse STEP:RATE milestones and enforce a true non-increasing decay."""
    points = []
    previous_step = 0
    previous_rate = initial_rate
    for value in values:
        try:
            step_text, rate_text = value.split(":", 1)
            step = int(step_text)
            rate = float(rate_text)
        except (TypeError, ValueError):
            raise ValueError(
                "learning rate schedule entries must use STEP:RATE, got %r" %
                value)
        if step <= previous_step:
            raise ValueError(
                "learning rate schedule steps must be positive and strictly "
                "increasing")
        if max_steps is not None and step > max_steps:
            raise ValueError(
                "learning rate schedule steps cannot exceed max steps")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("scheduled learning rates must be positive")
        if rate > previous_rate:
            raise ValueError(
                "learning rate schedule must be non-increasing")
        points.append((step, rate))
        previous_step = step
        previous_rate = rate
    return points


def optimize(session, graph, replay_memory, args, learning_rate):
    if len(replay_memory) < args.batch_size:
        return None
    minibatch = random.sample(replay_memory, args.batch_size)
    states, actions, rewards, next_states, terminals = zip(*minibatch)
    states = np.asarray(states, dtype=np.float32)
    next_states = np.asarray(next_states, dtype=np.float32)

    target_next_q = session.run(
        graph["target_q"],
        feed_dict={graph["target_state"]: next_states})
    if args.algorithm == "ddqn":
        online_next_q = session.run(
            graph["online_q"],
            feed_dict={graph["online_state"]: next_states})
    else:
        online_next_q = target_next_q
    target_values = calculate_targets(
        args.algorithm, rewards, terminals, online_next_q, target_next_q,
        args.gamma)

    loss_value, _ = session.run(
        [graph["loss"], graph["soft_update"]],
        feed_dict={
            graph["online_state"]: states,
            graph["actions"]: actions,
            graph["targets"]: target_values,
            graph["learning_rate"]: learning_rate,
        })
    return float(loss_value)


def train(args, game):
    run_directory, checkpoint_directory, log_path = run_paths(args.run_name)
    step_checkpoint_directory = os.path.join(
        run_directory, "step_checkpoints")
    existing_checkpoint = tf.train.get_checkpoint_state(checkpoint_directory)
    config_path = os.path.join(run_directory, RUN_CONFIG_FILENAME)
    has_existing_artifacts = bool(
        existing_checkpoint or os.path.isfile(log_path))
    if args.from_scratch and (
            has_existing_artifacts or os.path.isfile(config_path)):
        raise ValueError(
            "Run already exists. Choose a new --run-name for a fresh run.")

    os.makedirs(run_directory, exist_ok=True)
    ensure_run_config(args, run_directory, has_existing_artifacts)
    os.makedirs(checkpoint_directory, exist_ok=True)
    if args.step_checkpoint_interval:
        os.makedirs(step_checkpoint_directory, exist_ok=True)
    graph = create_training_graph(args)
    saver = tf.train.Saver(max_to_keep=5)
    step_saver = tf.train.Saver(max_to_keep=100)
    replay_memory = deque(maxlen=args.memory_size)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        checkpoint = None if args.from_scratch else existing_checkpoint
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print("Loaded checkpoint:", checkpoint.model_checkpoint_path)
        else:
            session.run(graph["hard_update"])
            print("Training from random weights")

        completed_episode, timestep = load_log_position(log_path)
        append_log = bool(checkpoint) and os.path.isfile(log_path)
        log_file = open(log_path, "a" if append_log else "w", newline="")
        writer = csv.DictWriter(log_file, fieldnames=TRAIN_LOG_FIELDS)
        if not append_log:
            writer.writeheader()
            log_file.flush()

        epsilon = epsilon_at_timestep(args, timestep)
        learning_rate = learning_rate_at_timestep(args, timestep)
        game_state = game.GameState()
        interrupted = False

        try:
            for episode in range(completed_episode + 1, args.episodes + 1):
                state = game_state.get_state()
                terminal = False
                score = 0
                episode_reward = 0.0
                episode_frames = 0
                episode_q_values = []
                episode_losses = []
                truncated = False

                while not terminal:
                    epsilon = epsilon_at_timestep(args, timestep)
                    learning_rate = learning_rate_at_timestep(args, timestep)
                    action, q_values = select_action(
                        session, graph, state, epsilon)
                    action_vector = np.zeros(ACTION_DIM, dtype=np.float32)
                    action_vector[action] = 1.0
                    _, reward, terminal = game_state.frame_step(
                        action_vector, return_image=args.render)
                    next_state = game_state.get_state()
                    replay_memory.append(
                        (state, action, reward, next_state, terminal))

                    # ``timestep`` counts completed environment transitions.
                    # Increment it before the update test so a 10,000-step
                    # warm-up performs its first update after transition
                    # 10,000, then every ``train_frequency`` transitions.
                    timestep += 1
                    learning_rate = learning_rate_at_timestep(args, timestep)
                    loss_value = None
                    if (
                            timestep >= args.learning_starts
                            and (timestep - args.learning_starts)
                            % args.train_frequency == 0):
                        loss_value = optimize(
                            session, graph, replay_memory, args,
                            learning_rate)
                    if loss_value is not None:
                        episode_losses.append(loss_value)
                    if q_values is not None:
                        episode_q_values.append(float(np.max(q_values)))

                    if (
                            args.step_checkpoint_interval
                            and timestep % args.step_checkpoint_interval == 0):
                        step_path = step_saver.save(
                            session,
                            os.path.join(
                                step_checkpoint_directory,
                                "state-" + args.algorithm),
                            global_step=timestep,
                            write_meta_graph=False,
                            write_state=False)
                        print(
                            "STEP CHECKPOINT", timestep,
                            "/ PATH", step_path)

                    state = next_state
                    episode_frames += 1
                    episode_reward += reward
                    if reward == 1:
                        score += 1
                    if args.max_steps and timestep >= args.max_steps:
                        truncated = not terminal
                        terminal = True

                completed_episode = episode
                writer.writerow({
                    "algorithm": args.algorithm,
                    "representation": "state4",
                    "seed": args.seed,
                    "timestep": timestep,
                    "episode": episode,
                    "score": score,
                    "episode_reward": episode_reward,
                    "episode_frames": episode_frames,
                    "epsilon": epsilon,
                    "learning_rate": learning_rate,
                    "mean_max_q": (
                        np.mean(episode_q_values) if episode_q_values else ""),
                    "mean_loss": (
                        np.mean(episode_losses) if episode_losses else ""),
                    "truncated": int(truncated),
                })
                log_file.flush()

                if episode % args.checkpoint_interval == 0:
                    saver.save(
                        session,
                        os.path.join(
                            checkpoint_directory,
                            "state-" + args.algorithm),
                        global_step=episode,
                        write_meta_graph=False)
                if episode % 10 == 0:
                    print(
                        "EPISODE", episode,
                        "/ TIMESTEP", timestep,
                        "/ SCORE", score,
                        "/ REWARD", round(episode_reward, 2),
                        "/ EPSILON", round(epsilon, 4),
                        "/ LEARNING RATE", format(learning_rate, ".8g"),
                        "/ LOSS", (round(np.mean(episode_losses), 6)
                                   if episode_losses else "warming-up"))
                if args.max_steps and timestep >= args.max_steps:
                    break
        except KeyboardInterrupt:
            interrupted = True
            print("Training interrupted; saving the latest network")
        finally:
            saver.save(
                session,
                os.path.join(checkpoint_directory, "state-" + args.algorithm),
                global_step=completed_episode,
                write_meta_graph=False)
            log_file.close()

        print(
            "Stopped" if interrupted else "Completed",
            args.algorithm.upper(), "run at episode", completed_episode,
            "and timestep", timestep)
        print("CSV:", log_path)


def collect_greedy_evaluation(
        session, graph, game, args, training_timestep=None):
    """Evaluate one restored policy on reproducible independent episodes."""
    scores = []
    durations = []
    evaluation_rows = []
    for episode in range(1, args.eval_episodes + 1):
        # Every checkpoint receives exactly the same independent test cases.
        eval_episode_seed = args.seed + episode - 1
        random.seed(eval_episode_seed)
        np.random.seed(eval_episode_seed)
        if hasattr(game, "PLAYER_INDEX_GEN"):
            game.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
        game_state = game.GameState()
        state = game_state.get_state()
        terminal = False
        score = 0
        frames = 0
        while not terminal and frames < args.eval_max_frames:
            q_values = session.run(
                graph["online_q"],
                feed_dict={graph["online_state"]: [state]})[0]
            action_vector = np.zeros(ACTION_DIM, dtype=np.float32)
            action_vector[int(np.argmax(q_values))] = 1.0
            _, reward, terminal = game_state.frame_step(
                action_vector, return_image=args.render)
            state = game_state.get_state()
            frames += 1
            if reward == 1:
                score += 1
        truncated = not terminal and frames >= args.eval_max_frames
        row = {
            "algorithm": args.algorithm,
            "run_name": args.run_name,
            "evaluation_seed": args.seed,
            "eval_episode_seed": eval_episode_seed,
            "episode": episode,
            "score": score,
            "frames": frames,
            "truncated": int(truncated),
        }
        if training_timestep is not None:
            row["training_timestep"] = training_timestep
        evaluation_rows.append(row)
        scores.append(score)
        durations.append(frames)
        checkpoint_text = (
            " / TRAINING TIMESTEP " + str(training_timestep)
            if training_timestep is not None else "")
        print(
            "EVAL EPISODE", episode,
            "/ SEED", eval_episode_seed,
            "/ SCORE", score,
            "/ FRAMES", frames,
            "/ TRUNCATED", int(truncated),
            checkpoint_text)
    return evaluation_rows, scores, durations


def evaluate(args, game):
    run_directory, checkpoint_directory, _ = run_paths(args.run_name)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_directory)
    if not checkpoint or not checkpoint.model_checkpoint_path:
        raise FileNotFoundError(
            "No checkpoint found for run " + args.run_name)

    graph = create_training_graph(args)
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint.model_checkpoint_path)
        evaluation_rows, scores, durations = collect_greedy_evaluation(
            session, graph, game, args)

    evaluation_path = os.path.join(
        run_directory, "evaluation_seed%d.csv" % args.seed)
    with open(evaluation_path, "w", newline="") as evaluation_file:
        writer = csv.DictWriter(evaluation_file, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    print("Checkpoint:", checkpoint.model_checkpoint_path)
    print(
        "Evaluation score: mean=%.3f std=%.3f max=%d" % (
            np.mean(scores), np.std(scores), np.max(scores)))
    print("Evaluation frames: mean=%.1f" % np.mean(durations))
    print("Evaluation CSV:", evaluation_path)


def evaluate_curve(args, game):
    """Evaluate fixed-timestep checkpoints with one shared greedy test suite."""
    run_directory, _, _ = run_paths(args.run_name)
    step_checkpoint_directory = os.path.join(
        run_directory, "step_checkpoints")
    if args.eval_timesteps:
        timesteps = list(args.eval_timesteps)
    else:
        timesteps = list(range(
            args.step_checkpoint_interval,
            args.max_steps + 1,
            args.step_checkpoint_interval))
    checkpoints = []
    missing = []
    for timestep in timesteps:
        checkpoint_path = os.path.join(
            step_checkpoint_directory,
            "state-%s-%d" % (args.algorithm, timestep))
        if os.path.isfile(checkpoint_path + ".index"):
            checkpoints.append((timestep, checkpoint_path))
        else:
            missing.append(checkpoint_path + ".index")
    if missing:
        raise FileNotFoundError(
            "Missing fixed-timestep checkpoints:\n" + "\n".join(missing))

    graph = create_training_graph(args)
    saver = tf.train.Saver()
    all_rows = []
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for timestep, checkpoint_path in checkpoints:
            saver.restore(session, checkpoint_path)
            print("Evaluating checkpoint:", checkpoint_path)
            rows, scores, durations = collect_greedy_evaluation(
                session, graph, game, args, training_timestep=timestep)
            all_rows.extend(rows)
            print(
                "TIMESTEP %d evaluation: mean=%.3f std=%.3f max=%d "
                "frames_mean=%.1f" % (
                    timestep, np.mean(scores), np.std(scores),
                    np.max(scores), np.mean(durations)))

    evaluation_path = os.path.join(
        run_directory, "evaluation_curve_seed%d.csv" % args.seed)
    with open(evaluation_path, "w", newline="") as evaluation_file:
        writer = csv.DictWriter(
            evaluation_file, fieldnames=STEP_EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print("Evaluation curve CSV:", evaluation_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Four-state-feature DQN/Double DQN for Flappy Bird.")
    parser.add_argument(
        "--mode", choices=("train", "eval", "eval-curve"),
        default="train")
    parser.add_argument(
        "--algorithm", choices=("dqn", "ddqn"), default="dqn")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-max-frames", type=int, default=10000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument(
        "--learning-starts", type=int, default=10000,
        help="collect this many transitions before gradient updates")
    parser.add_argument(
        "--train-frequency", type=int, default=4,
        help="perform one gradient update per this many environment steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--learning-rate-schedule", nargs="*", default=[],
        metavar="STEP:RATE",
        help=("piecewise-constant learning-rate decay milestones, for "
              "example 500000:0.00005 1000000:0.00001"))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--epsilon-start", type=float, default=0.1)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=200000,
        help="linearly anneal epsilon over this many environment steps")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--step-checkpoint-interval", type=int, default=0,
        help=("save permanent checkpoints at this timestep interval; "
              "required by eval-curve"))
    parser.add_argument(
        "--eval-timesteps", nargs="+", type=int,
        help=("evaluate only these fixed-timestep checkpoints in eval-curve "
              "mode; defaults to every --step-checkpoint-interval"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    args.learning_rate_schedule_points = parse_learning_rate_schedule(
        args.learning_rate_schedule, args.learning_rate, args.max_steps)
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.eval_episodes <= 0:
        raise ValueError("eval episodes must be positive")
    if args.eval_max_frames <= 0:
        raise ValueError("eval max frames must be positive")
    if args.hidden_size <= 0:
        raise ValueError("hidden size must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max steps must be positive when provided")
    if args.step_checkpoint_interval < 0:
        raise ValueError("step checkpoint interval must be non-negative")
    if args.step_checkpoint_interval:
        if args.max_steps is None:
            raise ValueError(
                "max steps is required with step checkpoint intervals")
        if args.max_steps % args.step_checkpoint_interval != 0:
            raise ValueError(
                "max steps must be divisible by step checkpoint interval")
    if args.eval_timesteps:
        if args.mode != "eval-curve":
            raise ValueError(
                "eval timesteps are only valid in eval-curve mode")
        if any(timestep <= 0 for timestep in args.eval_timesteps):
            raise ValueError("eval timesteps must be positive")
        if args.eval_timesteps != sorted(set(args.eval_timesteps)):
            raise ValueError(
                "eval timesteps must be unique and strictly increasing")
        if (args.max_steps is not None
                and args.eval_timesteps[-1] > args.max_steps):
            raise ValueError("eval timesteps cannot exceed max steps")
        if (args.step_checkpoint_interval
                and any(timestep % args.step_checkpoint_interval
                        for timestep in args.eval_timesteps)):
            raise ValueError(
                "eval timesteps must align with step checkpoint interval")
    if (args.mode == "eval-curve" and not args.step_checkpoint_interval
            and not args.eval_timesteps):
        raise ValueError(
            "eval-curve requires eval timesteps or a positive step "
            "checkpoint interval")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if args.epsilon_decay_steps <= 0:
        raise ValueError("epsilon decay steps must be positive")
    if args.learning_starts < 0:
        raise ValueError("learning starts must be non-negative")
    if args.train_frequency <= 0:
        raise ValueError("train frequency must be positive")
    if not 0 < args.tau <= 1:
        raise ValueError("tau must be in (0, 1]")

    if not args.render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sys.path.append("game/")
    import wrapped_flappy_bird as game

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    game.FPS = args.fps

    if args.mode == "eval":
        evaluate(args, game)
    elif args.mode == "eval-curve":
        evaluate_curve(args, game)
    else:
        train(args, game)


if __name__ == "__main__":
    main()
