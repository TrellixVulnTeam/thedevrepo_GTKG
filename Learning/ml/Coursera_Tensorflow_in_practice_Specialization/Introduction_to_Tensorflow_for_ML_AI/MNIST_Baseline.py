import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.99):
            print("\nExceeding accuracy of 99$...stopping training...\n")
            self.model.stop_training = True

(X_train, y_train), (X_test, y_test) = mnist.load_data()
EPOCHS = 10
IMG_DIMS = (28,28)
NUM_CLS = 10

if tf.keras.backend.image_data_format() == 'channel_first':
    X_train = X_train.reshape(X_train.shape[0], 1, IMG_DIMS[0], IMG_DIMS[1])
    X_test = X_test.reshape(X_test.shape[0], 1, IMG_DIMS[0], IMG_DIMS[1])
    input_shape = (1, IMG_DIMS[0], IMG_DIMS[1])
else:
    X_train = X_train.reshape(X_train.shape[0], IMG_DIMS[0], IMG_DIMS[1], 1)
    X_test = X_test.reshape(X_test.shape[0], IMG_DIMS[0], IMG_DIMS[1], 1)
    input_shape = (IMG_DIMS[0], IMG_DIMS[1], 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

print("Xs shape: {shape}".format(shape=X_train.shape))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(NUM_CLS, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

custom_cb = CustomCallback()

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, validation_data=(X_test, y_test), callbacks=[custom_cb])

