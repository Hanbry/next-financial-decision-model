from __future__ import absolute_import, division, print_function

import base64
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from datetime import datetime

import random
import string
import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from environments.financial_environment import FinancialEnvironment, create_environment

import models.qrnn_model as qrnn_model
import agents.agent_dqn as agent_dqn

# Hyperparameters
num_episodes = 200

collect_steps_per_iteration = 1
replay_buffer_max_length = 10000

batch_size = 32
learning_rate = 1e-3

window_size = 100
maximum_steps = 3000

log_interval = 200
eval_interval = maximum_steps*10
checkpoint_interval = maximum_steps*10
num_eval_episodes = 2

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        policy_state = policy.get_initial_state(batch_size=1)

        buy_count = 0
        sell_count = 0
        hold_count = 0

        while not time_step.is_last():
            action_step = policy.action(time_step, policy_state)
            if action_step.action == 0:
                buy_count += 1
            elif action_step.action == 1:
                sell_count += 1
            elif action_step.action == 2:
                hold_count += 1

            policy_state = action_step.state
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

        # print('episode return = {0}; buy = {1}; sell = {2}; hold = {3}'.format(episode_return, buy_count, sell_count, hold_count))

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_step(env, policy, policy_state, buffer):
    time_step = env.current_time_step()

    action_step = policy.action(time_step, policy_state)
    next_time_step = env.step(action_step.action)

    policy_state = action_step.state

    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)

    return (next_time_step, policy_state, action_step.action)

def collect_data(env, policy, buffer, steps, random):
    next_policy_state = policy.get_initial_state(batch_size=1)

    for i in range(steps):
        (next_time_step, next_policy_state, action) = collect_step(env, policy, next_policy_state, buffer)
    
        if next_time_step.is_last():
            env.reset()
            if not random:
                policy_state = policy.get_initial_state(batch_size=1)
            break

    return next_time_step, action

def train(data):

    log_dir = "logs/tensorbord/"
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    # Create Environments
    train_py_env, eval_py_env, train_env, eval_env = create_environment(data, window_size)

    # Create Model
    q_net = qrnn_model.create_model(train_py_env)

    target_q_net = qrnn_model.create_model(train_py_env)
    target_q_net.set_weights(q_net.get_weights())

    # Create Agent
    agent = agent_dqn.create_agent(train_env, q_net, target_q_net, learning_rate)

    # Create the replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    # Sample a batch of data from the buffer and update the agent's network.
    # Use `as_dataset()` to convert the replay buffer to a dataset and `iter()` to create an iterator
    # over the dataset.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.AUTOTUNE,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(tf.data.AUTOTUNE)

    iterator = iter(dataset)

    # Create Random Policy and pre-fill buffer with 100 steps
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
    random_policy.automatic_state_reset = True
    print()
    print("Prefill Buffer with random policy")
    print()
    collect_data(train_env, random_policy, replay_buffer, steps=100, random=True)

    # Training the agent
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    print("Calculate initial avg return")
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)

    returns = [avg_return]

    # Reset the environment.
    _ = train_env.reset()

    # Setup checkpointer
    train_checkpointer = common.Checkpointer(
        ckpt_dir="./checkpoints",
        max_to_keep=3,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step = agent.train_step_counter
    )

    print()
    print("Start Training")
    print()


    for episode_num in range(num_episodes):
        # metrics
        episode_avg_reward = 0
        steps_per_episode = 0
        action_histogram = np.zeros(train_env.action_spec().maximum + 1)

        while True:
            time_step, action = collect_data(train_env, agent.collect_policy, replay_buffer, steps=1, random=False)
            if time_step.is_last():
                break
            
            # Sample a batch of data from the buffer and update the agent's network.

            step = agent.train_step_counter.numpy()
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            # Update metrics
            if steps_per_episode == 0:
                episode_avg_reward = time_step.reward.numpy()[0]
            else:
                episode_avg_reward += time_step.reward.numpy()[0]/steps_per_episode

            steps_per_episode += 1
            action_histogram[action] += 1
            action_histogram_tensor = tf.convert_to_tensor(action_histogram, dtype=tf.float32)

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))
            if step % eval_interval == 0:
                print()
                print("START EVALUATION")
                print("+++++++++++++++++++++++")
                avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                print('Average Return = {0}'.format(avg_return))
                print("+++++++++++++++++++++++")
                returns.append(avg_return)
            if step % checkpoint_interval == 0:
                print()
                print("CHECKPOINT - SAVE MODEL")
                print()
                train_checkpointer.save(agent.train_step_counter.numpy())
                
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=agent.train_step_counter.numpy())
                tf.summary.scalar('average_return', avg_return, step=agent.train_step_counter.numpy())
                tf.summary.scalar('episode_reward', episode_avg_reward, step=episode_num)
                tf.summary.scalar('steps_per_episode', steps_per_episode, step=episode_num)
                tf.summary.histogram('actions_histogram', action_histogram_tensor, step=episode_num)
                train_summary_writer.flush()

        train_env.render(mode='human')

        

    print()
    print("START EVALUATION")
    print("+++++++++++++++++++++++")
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('Average Return = {0}'.format(avg_return))
    print("+++++++++++++++++++++++")
    print()
    returns.append(avg_return)
    print("Done")
    # Visualization
    plt.plot(returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim(top=250)
    plt.show()