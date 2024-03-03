import torch
import torchvision
from torchvision import datasets, transforms
import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from keras.optimizers import Adam
from keras.utils import to_categorical



# gpus = keras.config.experimental.list_physical_devices('GPU')

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)






      


data = tf.keras.utils.image_dataset_from_directory( '/Users/jedwoods/personal_projects/new_delicious/New-AIA/bev_classification/images/train_0', labels='inferred', label_mode='int', class_names=None, color_mode='rgb', batch_size=32, image_size=(128, 128), shuffle=True, seed=123, validation_split=None, subset=None, interpolation='bilinear', follow_links=False)

data = data.map(lambda x, y: (x/255, y))
data_iter = data.as_numpy_iterator()
batch = data_iter.next()
print(batch[1])

#partition the data into train, validation, and test sets
train_images = int(len(data) * 0.7)
val_images = int(len(data) * 0.2)+1
test_images = int(len(data) * 0.1)+1

train = data.take(train_images)
val = data.skip(train_images).take(val_images)
test = data.skip(val_images+train_images).take(val_images)

# train = train.map(lambda x, y: (x, to_categorical(y, num_classes=len(data.class_names))))
# val = val.map(lambda x, y: (x, to_categorical(y, num_classes=len(data.class_names))))
# test = test.map(lambda x, y: (x, to_categorical(y, num_classes=len(data.class_names))))

#CNN model
model = Sequential()
#first layer with 16 filters, each filter being a 3x3 matrix
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 256, 3)))
model.add(MaxPooling2D())

#second layer with 32 filters, each filter being a 3x3 matrix
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

#dense layer with 128 neurons
model.add(Dense(128, activation='relu'))

#output layer
model.add(Dense(1, activation='softmax'))

# calculate loss and accuracy
model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy(), metrics=['accuracy'])

logdir = "logs"

#tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

y_train = []
for images, labels in data:
    y_train.extend(labels.numpy())

# y_train_one_hot = to_categorical(y_train, num_classes=len(data.class_names))


hist = model.fit(train, validation_data=val, epochs=20, callbacks=[tensorboard_callback])

y_pred_prob = model.predict(test)

y_pred = y_pred_prob.argmax(axis=1)

test_labels = []
for images, labels in test:
    test_labels.extend(labels.numpy())

accuracy = (y_pred == test_labels).mean()
