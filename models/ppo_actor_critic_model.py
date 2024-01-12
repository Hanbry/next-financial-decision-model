import tensorflow as tf
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_rnn_network

def create_model(env):
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        env.observation_spec(),
        env.action_spec(),
        input_fc_layer_params = (100, 50),
        output_fc_layer_params = None,
        lstm_size = (50, )
    )
    value_net = value_rnn_network.ValueRnnNetwork(
        env.observation_spec(),
        input_fc_layer_params = (100, 50),
        output_fc_layer_params = None
    )

    return (actor_net, value_net)