"""

a. The build() method should define two trainable weights α and β, both of shape input_shape[-1:]
and data type tf.float32. α should be initialized with 1s, and β with 0s.

b. The call() method should compute the mean μ and standard deviation σ of each instance’s features. For this,
you can use tf.nn.moments(inputs, axes=-1, keepdims=True) , which returns the mean μ
and the variance σ 2 of all instances (compute the square root of the variance to get the standard deviation). Then the
function should compute and return α⊗(X - μ)/(σ + ε) + β, where ⊗ represents itemwise multiplication ( * ) and ε
is a smoothing term (small constant to avoid division by zero, e.g., 0.001).

c. Ensure that your custom layer produces the same (or very nearly the same) output as the
keras.layers.LayerNormalization layer.

"""
import numpy as np
import tensorflow as tf


class CustomLayerNormalization(tf.keras.layers.Layer):

    def __init__(self, eps=0.001, **kwargs):
        super(CustomLayerNormalization, self).__init__(**kwargs)
        self.eps = eps

    def build(self, batch_input_shape):
        self.alpha = self.add_weight(name="alpha", shape=batch_input_shape[-1:], dtype=tf.float32, initializer="ones")
        self.beta = self.add_weight(name="beta", shape=batch_input_shape[-1:], dtype=tf.float32, initializer="zeros")

        super(CustomLayerNormalization, self).build(batch_input_shape)

    def call(self, X):
        mean, var = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / (tf.sqrt(var) + self.eps)

    def compute_output_shape(self, batch_output_shape):
        return batch_output_shape

    def get_config(self):
        base_config = super(CustomLayerNormalization, self).get_config()
        return {**base_config, "eps": self.eps}


## Sample data

(X_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype(np.float32) / 255.

custom_layer_norm = CustomLayerNormalization()
keras_layer_norm = tf.keras.layers.LayerNormalization()

# print(keras_layer_norm(X_train).shape) # (60000, 28, 28)
# print(custom_layer_norm(X_train).shape) # (60000, 28, 28)
# print(tf.keras.losses.mean_absolute_error(keras_layer_norm(X_train), custom_layer_norm(X_train)).shape) # (60000, 28)
"""
To compare the custom layer normalization with the in-built one, it's applied first on a sample training set 
of shape (60000, 28,28). Then using MAE, the difference is computed whose mean is taken to quantify the difference.

Expected result is a very small value reflecting the closeness.
"""
result = tf.reduce_mean(tf.keras.losses.mean_absolute_error(keras_layer_norm(X_train), custom_layer_norm(X_train)))

print(result)
