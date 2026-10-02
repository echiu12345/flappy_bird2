#!/usr/bin/env python
"""Train and evaluate DQN/Double DQN agents from four numeric state features."""

import argparse
import csv
import os
import random
import sys
from collections import deque

import numpy as np
import tensorflow

tf = tensorflow.compat.v1
tf.disable_v2_behavior()

STATE_DIM = 4
ACTION_DIM = 2
STATE_EXPERIMENT_DIR = "state_experiments"


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

    optimizer = tf.train.AdamOptimizer(args.learning_rate)
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


def optimize(session, graph, replay_memory, args):
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
        })
    return float(loss_value)


def train(args, game):
    run_directory, checkpoint_directory, log_path = run_paths(args.run_name)
    existing_checkpoint = tf.train.get_checkpoint_state(checkpoint_directory)
    if args.from_scratch and (
            existing_checkpoint or os.path.isfile(log_path)):
        raise ValueError(
            "Run already exists. Choose a new --run-name for a fresh run.")

    os.makedirs(checkpoint_directory, exist_ok=True)
    graph = create_training_graph(args)
    saver = tf.train.Saver(max_to_keep=5)
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
        fieldnames = [
            "algorithm", "representation", "seed", "timestep", "episode",
            "score", "episode_reward", "episode_frames", "epsilon",
            "mean_max_q", "mean_loss",
        ]
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if not append_log:
            writer.writeheader()
            log_file.flush()

        epsilon = max(
            args.epsilon_end,
            args.epsilon_start * (args.epsilon_decay ** completed_episode))
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

                while not terminal:
                    action, q_values = select_action(
                        session, graph, state, epsilon)
                    action_vector = np.zeros(ACTION_DIM, dtype=np.float32)
                    action_vector[action] = 1.0
                    _, reward, terminal = game_state.frame_step(
                        action_vector, return_image=args.render)
                    next_state = game_state.get_state()
                    replay_memory.append(
                        (state, action, reward, next_state, terminal))

                    loss_value = optimize(
                        session, graph, replay_memory, args)
                    if loss_value is not None:
                        episode_losses.append(loss_value)
                    if q_values is not None:
                        episode_q_values.append(float(np.max(q_values)))

                    state = next_state
                    timestep += 1
                    episode_frames += 1
                    episode_reward += reward
                    if reward == 1:
                        score += 1
                    if args.max_steps and timestep >= args.max_steps:
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
                    "mean_max_q": (
                        np.mean(episode_q_values) if episode_q_values else ""),
                    "mean_loss": (
                        np.mean(episode_losses) if episode_losses else ""),
                })
                log_file.flush()

                epsilon = max(
                    args.epsilon_end, epsilon * args.epsilon_decay)
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


def evaluate(args, game):
    _, checkpoint_directory, _ = run_paths(args.run_name)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_directory)
    if not checkpoint or not checkpoint.model_checkpoint_path:
        raise FileNotFoundError(
            "No checkpoint found for run " + args.run_name)

    graph = create_training_graph(args)
    saver = tf.train.Saver()
    scores = []
    durations = []
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint.model_checkpoint_path)
        game_state = game.GameState()
        for episode in range(1, args.eval_episodes + 1):
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
            scores.append(score)
            durations.append(frames)
            print("EVAL EPISODE", episode, "/ SCORE", score, "/ FRAMES", frames)

    print("Checkpoint:", checkpoint.model_checkpoint_path)
    print(
        "Evaluation score: mean=%.3f std=%.3f max=%d" % (
            np.mean(scores), np.std(scores), np.max(scores)))
    print("Evaluation frames: mean=%.1f" % np.mean(durations))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Four-state-feature DQN/Double DQN for Flappy Bird.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--epsilon-start", type=float, default=0.1)
    parser.add_argument("--epsilon-end", type=float, default=0.001)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0 < args.epsilon_decay <= 1:
        raise ValueError("epsilon decay must be in (0, 1]")
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
    else:
        train(args, game)


if __name__ == "__main__":
    main()
