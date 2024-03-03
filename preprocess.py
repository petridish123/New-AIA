import torch
import torchvision
from torchvision import datasets, transforms
import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


print('test')

# transform = transforms.Compose([transforms.Resize((256, 512)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5),
#                                                      (0.5, 0.5, 0.5))])
# dataset = datasets.ImageFolder(root='bev_classification', transform=transform)

data = tf.keras.utils.image_dataset_from_directory( 'bev_classification', labels='inferred', label_mode='int', class_names=None, color_mode='rgb', batch_size=32, image_size=(128, 256), shuffle=True, seed=123, validation_split=None, subset=None, interpolation='bilinear', follow_links=False)

data = data.map(lambda x, y: (x/255, y))
data_iter = data.as_numpy_iterator()
batch = data_iter.next()

#partition the data into train, validation, and test sets
train_images = int(len(data) * 0.7)
val_images = int(len(data) * 0.2)+1
test_images = int(len(data) * 0.1)+1

train = data.take(train_images)
val = data.skip(train_images).take(val_images)
test = train_images.skip(val_images+train_images).take(val_images)


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

hist = model.fit(train, validation_data=val, epochs=20, callbacks=[tensorboard_callback])
