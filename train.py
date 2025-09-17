# We are going to follow this tutorial on classifying images
# with tensorflow
#
# https://www.tensorflow.org/tutorials/images/classification

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np

# Read in the data.  80% for training, 20% for validation
data_dir = "images"

batch_size = 32
img_height = 180
img_width = 180
seed = 123

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# copy the model from the tutorial
class_names = train_ds.class_names
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes),
])


# Train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=40
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# Now go back and rerun the model on all of the images.

model2 = Sequential([model, layers.Softmax()])
for cl in range(5):
    print("================ Directory %d ==================" % (cl,))
    subdir = "%s/%d" % (data_dir, cl)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        subdir,
        label_mode = None,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    infer = model2.predict(test_ds)

    for x in infer:
        for v in x:
            print("%8.4f " % (v,), end="")
        print(np.argmax(x))
