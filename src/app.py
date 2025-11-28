import tensorflow as tf
import keras
from keras import layers
import tensorflow_datasets as tfds

training_data, training_info = tfds.load("mnist", split='train', shuffle_files=True, as_supervised=True, with_info=True)

def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label

training_data = training_data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
training_data = training_data.cache()
training_data = training_data.shuffle(training_info.splits['train'].num_examples)
training_data = training_data.batch(128)
training_data = training_data.prefetch(tf.data.AUTOTUNE)

print(training_data)

model = keras.Sequential([
  layers.Flatten(input_shape=(28, 28)),
  layers.Dense(128, activation='relu'),
  layers.Dense(10)
])
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    training_data,
    epochs=6,
)