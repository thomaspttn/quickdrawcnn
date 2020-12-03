from utils import generate_images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
CNN_MAIN.py
Author: Thomas Patton <tjp94@case.edu>
Class: ECSE 484 Final Project

This Python file controls the main training of the convolutional
net. See README.md for more information.
'''

# Un-comment to see GPU status for training
print(tf.config.list_physical_devices('GPU'))

# Set seed for reproducibility
tf.random.set_seed(999)

class_labels = ['bear', 'bee', 'bird', 'cat', 'dog', 'dolphin', 'fish',
                'horse', 'shark', 'tiger']
base_dir = './images'

# Un-comment this line if you do not already have the 28x28 images
# generate_images()

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Define our convolutional network
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), padding='same',
                           activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(len(class_labels), activation='softmax')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

print(model.summary())

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

spe = 200
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    batch_size=200,
    color_mode='grayscale',
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(28, 28),
    shuffle=False,
    batch_size=50,
    color_mode='grayscale',
    class_mode='categorical')

history = model.fit(
    train_generator,
    steps_per_epoch=spe,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=spe/4,
    verbose=2)

y_pred = np.rint(model.predict(validation_generator))
Y_pred = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

print(tf.math.confusion_matrix(y_true, Y_pred))
print(classification_report(y_true, Y_pred))
