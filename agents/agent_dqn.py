import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

def create_agent(env, q_net, target_q_net, learning_rate):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        target_q_network=target_q_net,
        target_update_period=1000,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()
    return agent