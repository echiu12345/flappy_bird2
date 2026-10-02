#!/usr/bin/env python
from __future__ import print_function

import argparse
import csv
import tensorflow

tf = tensorflow.compat.v1
tf.disable_v2_behavior()

import cv2
import os
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
TARGET_UPDATE_INTERVAL = 1000 # frames between target-network updates
DOUBLE_DQN_NETWORK_DIR = "saved_networks_double_dqn"
DQN_NETWORK_DIR = "saved_networks_dqn"
EXPERIMENT_DIR = "experiments"

def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial, trainable=trainable)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork(scope=None, trainable=True):
    if scope:
        with tf.variable_scope(scope):
            return createNetwork(trainable=trainable)

    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32], trainable)
    b_conv1 = bias_variable([32], trainable)

    W_conv2 = weight_variable([4, 4, 32, 64], trainable)
    b_conv2 = bias_variable([64], trainable)

    W_conv3 = weight_variable([3, 3, 64, 64], trainable)
    b_conv3 = bias_variable([64], trainable)

    W_fc1 = weight_variable([1600, 512], trainable)
    b_fc1 = bias_variable([512], trainable)

    W_fc2 = weight_variable([512, ACTIONS], trainable)
    b_fc2 = bias_variable([ACTIONS], trainable)

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    parameters = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3,
                  b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
    return s, readout, h_fc1, parameters

def build_target_update(online_parameters, target_parameters):
    return tf.group(*[
        target.assign(online)
        for online, target in zip(online_parameters, target_parameters)
    ])

def double_dqn_targets(rewards, terminals, online_next_q, target_next_q):
    """Calculate Double DQN targets using online selection and target evaluation."""
    rewards = np.asarray(rewards, dtype=np.float32)
    best_next_actions = np.argmax(online_next_q, axis=1)
    target_values = target_next_q[np.arange(len(best_next_actions)), best_next_actions]
    targets = rewards + GAMMA * target_values
    return np.where(np.asarray(terminals), rewards, targets).astype(np.float32)

def dqn_targets(rewards, terminals, target_next_q):
    """Calculate standard DQN targets using the target network for max Q."""
    rewards = np.asarray(rewards, dtype=np.float32)
    target_values = np.max(target_next_q, axis=1)
    targets = rewards + GAMMA * target_values
    return np.where(np.asarray(terminals), rewards, targets).astype(np.float32)

def resolve_run_paths(algorithm, run_name):
    if run_name:
        run_directory = os.path.join(EXPERIMENT_DIR, run_name)
        return (
            os.path.join(run_directory, "checkpoints"),
            os.path.join(run_directory, "episodes.csv"),
        )
    checkpoint_directory = (
        DOUBLE_DQN_NETWORK_DIR if algorithm == "ddqn" else DQN_NETWORK_DIR)
    return checkpoint_directory, os.path.join(
        "training_logs", algorithm + "_episodes.csv")

def last_logged_episode(log_path):
    if not os.path.isfile(log_path):
        return 0
    with open(log_path, "r", newline="") as log_file:
        rows = list(csv.DictReader(log_file))
    return int(rows[-1]["episode"]) if rows else 0

def trainNetwork(s, readout, h_fc1, online_parameters,
                 target_s, target_readout, target_parameters, sess, args):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(
        cost, var_list=online_parameters)
    update_target_network = build_target_update(
        online_parameters, target_parameters)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    checkpoint_directory, log_path = resolve_run_paths(
        args.algorithm, args.run_name)
    os.makedirs(checkpoint_directory, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.initialize_all_variables())
    starting_t = 0
    checkpoint = None if args.from_scratch else tf.train.get_checkpoint_state(
        checkpoint_directory)
    restored_experiment = checkpoint and checkpoint.model_checkpoint_path
    if restored_experiment:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        try:
            starting_t = int(checkpoint.model_checkpoint_path.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            starting_t = 0
        print("Successfully loaded", args.algorithm.upper() + ":",
              checkpoint.model_checkpoint_path)
    else:
        legacy_checkpoint = (
            None if args.from_scratch
            else tf.train.get_checkpoint_state("saved_networks"))
        if legacy_checkpoint and legacy_checkpoint.model_checkpoint_path:
            checkpoint_variables = dict(tf.train.list_variables(
                legacy_checkpoint.model_checkpoint_path))
            compatible_variables = {
                variable.op.name: variable
                for variable in tf.global_variables()
                if variable.op.name in checkpoint_variables
                and variable.shape.as_list() == checkpoint_variables[variable.op.name]
            }
            tf.train.Saver(var_list=compatible_variables).restore(
                sess, legacy_checkpoint.model_checkpoint_path)
            print("Successfully loaded legacy DQN:",
                  legacy_checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights; training from scratch")
        sess.run(update_target_network)

    append_log = starting_t > 0 and os.path.isfile(log_path)
    log_file = open(log_path, "a" if append_log else "w", newline="")
    fieldnames = [
        "algorithm", "seed", "timestep", "episode", "score",
        "episode_reward", "episode_frames", "epsilon", "mean_max_q",
        "mean_loss",
    ]
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    if not append_log:
        log_writer.writeheader()
        log_file.flush()

    # start training
    explored_frames = max(0, starting_t - args.observe)
    epsilon = max(
        args.final_epsilon,
        args.initial_epsilon -
        (args.initial_epsilon - args.final_epsilon) *
        explored_frames / args.explore)
    t = starting_t
    episode = last_logged_episode(log_path) if append_log else 0
    episode_score = 0
    episode_reward = 0.0
    episode_frames = 0
    episode_max_q = []
    episode_losses = []
    print("Algorithm:", args.algorithm.upper(), "/ Run:",
          args.run_name or "default", "/ Log:", log_path)
    while args.max_steps is None or t < args.max_steps:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > args.final_epsilon and t > args.observe:
            epsilon -= (
                args.initial_epsilon - args.final_epsilon) / args.explore

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        episode_reward += r_t
        episode_frames += 1
        episode_max_q.append(float(np.max(readout_t)))
        if r_t == 1:
            episode_score += 1
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > args.observe and len(D) >= BATCH:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            terminal_batch = [d[4] for d in minibatch]

            target_next_q = target_readout.eval(
                feed_dict={target_s: s_j1_batch})
            if args.algorithm == "ddqn":
                # Online selection and target-network evaluation.
                online_next_q = readout.eval(feed_dict={s: s_j1_batch})
                y_batch = double_dqn_targets(
                    r_batch, terminal_batch, online_next_q, target_next_q)
            else:
                # Standard DQN uses max Q from the target network.
                y_batch = dqn_targets(
                    r_batch, terminal_batch, target_next_q)

            # perform gradient step
            loss_value, _ = sess.run([cost, train_step], feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )
            episode_losses.append(float(loss_value))

        # update the old values
        s_t = s_t1
        t += 1

        if terminal:
            episode += 1
            log_writer.writerow({
                "algorithm": args.algorithm,
                "seed": args.seed,
                "timestep": t,
                "episode": episode,
                "score": episode_score,
                "episode_reward": episode_reward,
                "episode_frames": episode_frames,
                "epsilon": epsilon,
                "mean_max_q": np.mean(episode_max_q),
                "mean_loss": (np.mean(episode_losses)
                              if episode_losses else ""),
            })
            log_file.flush()
            episode_score = 0
            episode_reward = 0.0
            episode_frames = 0
            episode_max_q = []
            episode_losses = []

        if t % args.target_update_interval == 0:
            sess.run(update_target_network)
            print("Updated target network at timestep", t)

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, os.path.join(
                checkpoint_directory, GAME + '-' + args.algorithm),
                global_step=t, write_meta_graph=False)

        # print info
        state = ""
        if t <= args.observe:
            state = "observe"
        elif t > args.observe and t <= args.observe + args.explore:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))

    saver.save(sess, os.path.join(
        checkpoint_directory, GAME + '-' + args.algorithm),
        global_step=t, write_meta_graph=False)
    log_file.close()

def playGame(args):
    game.FPS = args.fps
    sess = tf.InteractiveSession()
    s, readout, h_fc1, online_parameters = createNetwork()
    target_s, target_readout, _, target_parameters = createNetwork(
        scope="target", trainable=False)
    trainNetwork(s, readout, h_fc1, online_parameters,
                 target_s, target_readout, target_parameters, sess, args)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Flappy Bird with DQN or Double DQN.")
    parser.add_argument(
        "--algorithm", choices=("dqn", "ddqn"), default="ddqn")
    parser.add_argument(
        "--run-name", help="Store checkpoints and CSV data under experiments/NAME.")
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Do not restore an existing or legacy checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--observe", type=int, default=int(OBSERVE))
    parser.add_argument("--explore", type=int, default=int(EXPLORE))
    parser.add_argument("--initial-epsilon", type=float)
    parser.add_argument("--final-epsilon", type=float, default=FINAL_EPSILON)
    parser.add_argument(
        "--target-update-interval", type=int,
        default=TARGET_UPDATE_INTERVAL)
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Game speed limit; use 0 for uncapped training.")
    parser.add_argument(
        "--max-steps", type=int,
        help="Stop and save after this environment timestep.")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.initial_epsilon is None:
        # Named experiment runs use the training epsilon both when starting
        # fresh and when resuming from their checkpoint.
        args.initial_epsilon = (
            0.1 if args.from_scratch or args.run_name else INITIAL_EPSILON)
    if args.explore <= 0 or args.target_update_interval <= 0:
        raise ValueError("explore and target-update-interval must be positive")
    if not 0 <= args.final_epsilon <= args.initial_epsilon <= 1:
        raise ValueError("epsilon values must satisfy 0 <= final <= initial <= 1")
    if args.fps < 0:
        raise ValueError("fps cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    playGame(args)

if __name__ == "__main__":
    main()
