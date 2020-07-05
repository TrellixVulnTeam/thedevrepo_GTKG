import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

epochs = 10
batch_size = 64
num_cls = 10
img_size = X_train[0].shape

if tf.keras.backend.image_data_format == 'channel_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_size[0], img_size[1])
    X_test = X_test.reshape(X_test.shape[0], 1, img_size[0], img_size[1])
    input_shape = (1, img_size[0], img_size[1])
else:
    X_train = X_train.reshape(X_train.shape[0], img_size[0], img_size[1], 1)
    X_test = X_test.reshape(X_test.shape[0], img_size[0], img_size[1], 1)
    input_shape = (img_size[0], img_size[1], 1)

X_train.astype('float32')
X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

fcn = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=img_size),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_cls, activation='softmax')
])

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_cls, activation='softmax')
])

fcn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= 0.99:
            self.model.stop_training = True
            print(
                "\nExceeding accuracy of {pc}%...stopping training at epoch #{e}\n".format(pc=logs.get('val_accuracy') * 100,
                                                                                           e=epoch))


custom_callback = CustomEarlyStopping()

fcn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
        callbacks=[custom_callback])
