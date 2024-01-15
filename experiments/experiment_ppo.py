import functools
import os
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics

from environments.advanced_financial_ppo_environment import create_environment
import agents.agent_ppo as agent_ppo
import models.ppo_actor_critic_model as ppo_actor_critic_model

# Configure hyperparameters
num_environment_steps = 25000000  # Number of training steps
collect_episodes_per_iteration = 2  # Number of episodes to collect per iteration
replay_buffer_capacity = 1001  # Replay buffer capacity
learning_rate = 1e-3  # Learning rate
num_parallel_environments = 1
num_epochs = 25
num_eval_episodes = 10
summary_interval = 50
eval_interval = 500
log_interval = 50
train_checkpoint_interval = 500
policy_checkpoint_interval = 500
summaries_flush_secs = 1
debug_summaries = False
summarize_grads_and_vars = False

def train(data):
    train_dir = "results/ppo/train"
    eval_dir = "results/ppo/eval"
    checkpoint_dir = "checkpoints/ppo"
    saved_model_dir = "results/ppo/policy_saved_model"

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis = summaries_flush_secs * 1000
    )
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis = summaries_flush_secs * 1000
    )
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Create Environments
    train_env, eval_env = create_environment(data, num_parallel_environments)

    # Create Model
    (actor_net, value_net) = ppo_actor_critic_model.create_model(train_env)

    # Create Agent
    agent = agent_ppo.create_agent(train_env, actor_net, value_net, learning_rate, num_epochs, global_step)

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(batch_size=num_parallel_environments),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=num_parallel_environments),
    ]

    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    # Replay Buffer for collected data
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_buffer_capacity
    )

    train_checkpointer = common.Checkpointer(
        ckpt_dir = checkpoint_dir,
        agent = agent,
        global_step = global_step,
        metrics = metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
    )

    policy_checkpointer = common.Checkpointer(
        ckpt_dir = os.path.join(checkpoint_dir, 'policy'),
        policy = eval_policy,
        global_step = global_step,
    )

    saved_model = policy_saver.PolicySaver(eval_policy, train_step = global_step)

    train_checkpointer.initialize_or_restore()

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env,
        collect_policy,
        observers = [replay_buffer.add_batch] + train_metrics,
        num_episodes = collect_episodes_per_iteration
    )

    def train_step():
        trajectories = replay_buffer.gather_all()
        return agent.train(experience=trajectories)
    
    collect_driver.run = common.function(collect_driver.run, autograph=False)
    agent.train = common.function(agent.train, autograph=False)
    train_step = common.function(train_step)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    while environment_steps_metric.result() < num_environment_steps:
        global_step_val = global_step.numpy()
        print("global step: ", global_step_val)

        if global_step_val % eval_interval == 0:
            print("evaluating policy")
            metric_utils.eager_compute(
                eval_metrics,
                eval_env,
                eval_policy,
                num_episodes = num_eval_episodes,
                train_step = global_step,
                summary_writer = eval_summary_writer,
                summary_prefix = 'Metrics'
            )
            print("evaluation complete")

        start_time = time.time()
        collect_driver.run()
        print("collected all data")
        collect_time += time.time() - start_time

        start_time = time.time()
        total_loss, _ = train_step()
        print("trained agent")
        replay_buffer.clear()
        train_time += time.time() - start_time

        for train_metric in train_metrics:
            train_metric.tf_summaries(train_step = global_step, step_metrics = step_metrics)

        if global_step_val % log_interval == 0:
            train_env.render()
            print('step =', global_step_val, 'loss =', total_loss.numpy())
            steps_per_sec = (global_step_val - timed_at_step) / (collect_time + train_time)
            print(steps_per_sec, ' steps/sec')
            print('collect_time = ', collect_time, 'train_time = ', train_time)

            with tf.compat.v2.summary.record_if(True):
                tf.compat.v2.summary.scalar(name = 'global_steps_per_sec', data = steps_per_sec, step = global_step)

            if global_step_val % train_checkpoint_interval == 0:
                train_checkpointer.save(global_step = global_step_val)

            if global_step_val % policy_checkpoint_interval == 0:
                policy_checkpointer.save(global_step = global_step_val)
                saved_model_path = os.path.join(saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
                saved_model.save(saved_model_path)

            timed_at_step = global_step_val
            collect_time = 0
            train_time = 0

    # One final eval before exiting.
    metric_utils.eager_compute(
        eval_metrics,
        eval_env,
        eval_policy,
        num_episodes = num_eval_episodes,
        train_step = global_step,
        summary_writer = eval_summary_writer,
        summary_prefix = 'Metrics'
    )