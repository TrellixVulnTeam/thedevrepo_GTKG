# Model Subclassing
import tensorflow as tf


class MultiInputMultiOutputModel(tf.keras.Model):
    def __init__(self, hidden_dims, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(units=hidden_dims[0], activation='relu')
        self.hidden2 = tf.keras.layers.Dense(units=hidden_dims[1], activation='relu')
        self.main_output = tf.keras.layers.Dense(units=1)
        self.aux_output = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        input_a, input_b = inputs
        hidden1 = self.hidden1(input_a)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([input_b, hidden2])
        main_out = self.main_output(concat)
        aux_out = self.aux_output(hidden2)

        return main_out, aux_out


sample_mimo_model = MultiInputMultiOutputModel()
