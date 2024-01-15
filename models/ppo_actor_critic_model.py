import tensorflow as tf
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_rnn_network

actor_fc_layers = [200, 100]
value_fc_layers = [200, 100]
lstm_size = [100, 100]

def create_model(env):
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        env.observation_spec(),
        env.action_spec(),
        input_fc_layer_params = actor_fc_layers,
        output_fc_layer_params = actor_fc_layers,
        lstm_size = lstm_size
    )
    value_net = value_rnn_network.ValueRnnNetwork(
        env.observation_spec(),
        input_fc_layer_params = value_fc_layers,
        output_fc_layer_params = actor_fc_layers,
        lstm_size = lstm_size
    )

    # actor_net.build()
    # value_net.build()

    # print('Actor Network Summary:')
    # print(actor_net.summary())
    # print()
    # print('Value Network Summary:')
    # print(value_net.summary())

    return (actor_net, value_net)