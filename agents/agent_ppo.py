import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent

def create_agent(env, actor_net, value_net, learning_rate, num_epochs, global_step):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    agent = ppo_agent.PPOAgent(
        env.time_step_spec(),
        env.action_spec(),
        optimizer,
        actor_net = actor_net,
        value_net = value_net,
        entropy_regularization = 0.0,
        normalize_observations = False,
        normalize_rewards = False,
        use_gae = True,
        num_epochs = num_epochs,
        # debug_summaries = debug_summaries,
        # summarize_grads_and_vars = summarize_grads_and_vars,
        train_step_counter = global_step
    )

    agent.initialize()
    return agent
