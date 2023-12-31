from matplotlib import pyplot as plt
from generator import DataGenerator
from loader import load
from models import cats_and_dogs
import tensorflow as tf

data_path = "/home/vadym/Downloads/archive"
weights_path = "cats.h5"

train_images1, train_labels1, valid_images1, valid_labels1 = load(data_path)

train_generator = DataGenerator(8, train_images1, train_labels1, shuffle=True)
valid_generator = DataGenerator(8, valid_images1, valid_labels1, shuffle=False)

model = cats_and_dogs((80, 80, 3))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(weights_path, save_best_only=True, period=1, save_weights_only=True)
]

history = model.fit(train_generator, validation_data=valid_generator, epochs=20, callbacks=callbacks)

plt.plot(history.history['loss'][20:])
plt.plot(history.history['val_loss'][20:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

