import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

fmnist = tf.keras.datasets.fashion_mnist

(X_train,y_train), (X_test, y_test) = fmnist.load_data()
print("Data shape: {}, {}".format(X_train.shape, y_train.shape))

X_train, X_val = X_train[5000:] / 255.0, X_train[:5000] / 255.0
y_train, y_val = y_train[5000:], y_train[:5000]

print("X_train has {} obs. X_val has {} obs. having shape: {}".format(X_train.shape[0], X_val.shape[0], X_train[0].shape))
img_shape = X_train[0].shape

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=img_shape))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_val, y_val), verbose=2)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.evaluate(X_test, y_test, verbose=2)
