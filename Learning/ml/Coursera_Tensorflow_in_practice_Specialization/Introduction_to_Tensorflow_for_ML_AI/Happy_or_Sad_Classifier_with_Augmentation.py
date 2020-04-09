import tensorflow as tf
import os
import zipfile


DESIRED_ACCURACY = 0.999

!wget --no-check-certificate \
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
    -O "/tmp/happy-or-sad.zip"

zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > DESIRED_ACCURACY):
      print('\nExceeds desired accuracy...stopping training...\n')
      self.model.stop_training = True

callbacks = myCallback()

# This Code Block should Define and Compile the Model
input_shape = (28, 28, 3)

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64,(3,3),activation='relu', input_shape=input_shape),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(2, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# This code block should create an instance of an ImageDataGenerator called train_datagen
# And a train_generator by calling train_datagen.flow_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_directory(
        '/tmp/h-or-s',
        target_size=(28,28))

# Expected output: 'Found 80 images belonging to 2 classes'

# This code block should call model.fit and train for
# a number of epochs.


history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=100,
    verbose=2,
    callbacks=[callbacks]
)

# Expected output: "Reached 99.9% accuracy so cancelling training!""