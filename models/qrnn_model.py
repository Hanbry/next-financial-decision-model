from tf_agents.specs import tensor_spec
from tf_agents.networks import q_rnn_network

def create_model(env):
    observation_tensor_spec = tensor_spec.from_spec(env.observation_spec())
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    observation_shape = observation_tensor_spec.shape
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define the Q-RNN Network
    lstm_size = (128, 64, 32)
    q_net = q_rnn_network.QRnnNetwork(
        env.observation_spec(),
        env.action_spec(),
        lstm_size=lstm_size)
    
    # fc_layer_params = (100, 50) # Schichten und Neuronenanzahl pro Schicht
    # q_net = q_network.QNetwork(
    #     env.observation_spec(),
    #     env.action_spec(),
    #     fc_layer_params=fc_layer_params)

    return q_net