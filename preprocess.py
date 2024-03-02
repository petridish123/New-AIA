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

train_images = int(len(data) * 0.7)
val_images = int(len(data) * 0.2)+1
test_images = int(len(data) * 0.1)+1

train_images = data.take(train_images)
val_images = data.skip(train_images).take(val_images)
test_images = train_images.skip(val_images+train_images).take(val_images)




