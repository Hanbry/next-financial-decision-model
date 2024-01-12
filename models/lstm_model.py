import tensorflow as tf
from tf_agents.networks import network

class NextFinancialNetwork(network.Network):

    def __init__(self, input_tensor_spec, output_tensor_spec, name='NextFinancialNetwork'):
        super(NextFinancialNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        # Convolutional layers
        self._conv_layers = [
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.Conv1D(64, 3, activation='relu')
        ]

        # LSTM layers
        self._lstm_layers = [
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50)
        ]

        # Dense layers
        self._dense_layers = [
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu')
        ]

        # Output layer
        self._output_layer = tf.keras.layers.Dense(
            output_tensor_spec.shape[0],
            activation='sigmoid')

    def call(self, observations, step_type=None, network_state=()):
        outputs = observations

        # Pass through the convolutional layers
        for layer in self._conv_layers:
            outputs = layer(outputs)

        # Pass through the LSTM layers
        for layer in self._lstm_layers:
            outputs = layer(outputs)

        # Pass through the dense layers
        for layer in self._dense_layers:
            outputs = layer(outputs)

        # Output layer
        outputs = self._output_layer(outputs)

        return outputs, network_state
