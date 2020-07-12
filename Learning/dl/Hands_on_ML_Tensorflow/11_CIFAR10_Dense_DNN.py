"""

Practice training a deep neural network on the CIFAR10 image
dataset

	a. Build a DNN with 20 hidden layers of 100 neurons each
	(that’s too many, but it’s the point of this exercise). Use
	He initialization and the ELU activation function.

	b. Using Nadam optimization and early stopping, train the
	network on the CIFAR10 dataset. You can load it with
	keras.datasets.cifar10.load_ data() . The dataset is
	composed of 60,000 32 × 32–pixel color images (50,000
	for training, 10,000 for testing) with 10 classes, so you’ll
	need a softmax output layer with 10 neurons. Remember
	to search for the right learning rate each time you change
	the model’s architecture or hyperparameters.

	c. Now try adding Batch Normalization and compare the
	learning curves: Is it converging faster than before? Does
	it produce a better model? How does it affect training
	speed?

	d. Try replacing Batch Normalization with SELU, and make
	the necessary adjustements to ensure the network self-
	normalizes (i.e., standardize the input features, use
	LeCun normal initialization, make sure the DNN contains
	only a sequence of dense layers, etc.).

	e. Try regularizing the model with alpha dropout. Then,
	without retraining your model, see if you can achieve
	better accuracy using MC Dropout.

	f. Retrain your model using 1cycle scheduling and see if it
	improves training speed and model accuracy.

"""

import tensorflow as tf

device_name = tf.test.gpu_device_name()
print(device_name)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
print("X_train.shape: {}, X_test.shape: {}".format(X_train.shape, X_test.shape))

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

img_dims = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

num_units = 20
num_layers = 100
batch_size = 32
epochs = 20

print(img_dims)

dnn_100_model = tf.keras.models.Sequential()
dnn_100_model.add(tf.keras.layers.Flatten(input_shape=img_dims))

for _ in range(num_layers):
    dnn_100_model.add(tf.keras.layers.Dense(units=num_units, activation='elu', kernel_initializer='he_uniform'))

dnn_100_model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
dnn_100_model.compile(optimizer=tf.keras.optimizers.Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

print(dnn_100_model.summary())

history = dnn_100_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                            verbose=2)
