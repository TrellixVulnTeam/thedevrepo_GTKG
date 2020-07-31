"""

Train a model using a custom training loop to tackle the Fashion
MNIST dataset (see Chapter 10).

a. Display the epoch, iteration, mean training loss, and
mean accuracy over each epoch (updated at each
iteration), as well as the validation loss and accuracy at
the end of each epoch.

"""

import numpy as np
import tensorflow as tf

(X, y), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X = X.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.

X_valid, X_train = X[:5000], X[5000:]
y_valid, y_train = y[:5000], y[5000:]

input_shape = (X.shape[1], X.shape[2])
np.random.seed(42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

epochs = 10
batch_size = 64
train_steps = len(X_train) // batch_size
optimizer = tf.keras.optimizers.Nadam(lr=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]


def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


for epoch in range(epochs):
    print("Epoch #{}".format(epoch + 1))
    for step in range(train_steps):
        # print("Training step #{}".format(step + 1))
        status = {}
        X_b, y_b = random_batch(X_train, y_train, batch_size)

        # tf.GradientTape is used for autodiff during backprop as this is a custom training loop.
        with tf.GradientTape() as tape:
            y_pred = model(X_b)  # output
            sparse_catxentropy_loss = tf.reduce_mean(loss(y_b, y_pred))  # calculate mean loss
            losses = tf.add_n([sparse_catxentropy_loss] + model.losses)  # ??

        grads = tape.gradient(losses, model.trainable_variables)  # calculate gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))  # apply gradients / updating parameters
        mean_loss(losses)
        status['loss'] = mean_loss.result().numpy()

        for metric in metrics:
            metric(y_b, y_pred)
            status[metric.name] = metric.result().numpy()

        y_pred_val = model(X_valid)  # output from validation set
        status['val_loss'] = np.mean(loss(y_valid, y_pred_val))  # val_loss
        status['val_acc'] = np.mean(
            tf.keras.metrics.sparse_categorical_accuracy(
                tf.constant(y_valid, dtype=np.float32), y_pred_val
            )
        )  # tf.constant is used because y_pred_val is a tensor ??

        print("Train loss: {:.2f}, Val loss: {:.2f} Val acc: {:.2f}".format(
            status['loss'], status['val_loss'], status['val_acc']))
    for metric in [mean_loss] + metrics:
        metric.reset_states()  # ??

tf.keras.backend.clear_session()

"""
b. Try using a different optimizer with a different learning rate for the upper layers and the lower layers.
"""

lower_layers = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(100, activation="relu")
])

upper_layers = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation="softmax")
])

model = tf.keras.models.Sequential([
    lower_layers, upper_layers
])
lower_optimizer = tf.keras.optimizers.SGD(lr=0.005)
upper_optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)

for epoch in range(epochs):
    print("Epoch #{}".format(epoch))
    for step in range(train_steps):
        X_b, y_b = random_batch(X_train, y_train, batch_size)
        status = {}

        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(X_b)
            sparse_catxentropy_loss = tf.reduce_mean(loss(y_b, y_pred))  # calculate mean loss
            losses = tf.add_n([sparse_catxentropy_loss] + model.losses)  # ??

        for layers, optimizer in (
                (upper_layers, upper_optimizer), (lower_layers, lower_optimizer)
        ):
            grads = tape.gradient(losses, layers.trainable_variables)
            optimizer.apply_gradients(zip(grads, layers.trainable_variables))

        del tape
        mean_loss(losses)
        status['loss'] = mean_loss.result().numpy()

        for metric in metrics:
            metric(y_b, y_pred)
            status[metric.name] = metric.result().numpy()

        y_pred_val = model(X_valid)
        status['val_loss'] = np.mean(loss(y_valid, y_pred_val))
        status['val_acc'] = np.mean(
            tf.keras.metrics.sparse_categorical_accuracy(
                tf.constant(y_valid, dtype=np.float32), y_pred_val
            )
        )

        print("Train loss: {:.2f}, Val loss: {:.2f} Val acc: {:.2f}".format(
            status['loss'], status['val_loss'], status['val_acc']))
    for metric in [mean_loss] + metrics:
        metric.reset_states()  # ??

tf.keras.backend.clear_session()
