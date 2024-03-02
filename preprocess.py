import torch
import torchvision
from torchvision import datasets, transforms
import keras
import tensorflow as tf
print('test')

# transform = transforms.Compose([transforms.Resize((256, 512)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5),
#                                                      (0.5, 0.5, 0.5))])
# dataset = datasets.ImageFolder(root='bev_classification', transform=transform)

data = tf.keras.utils.image_dataset_from_directory( 'bev_classification', labels='inferred', label_mode='int', class_names=None, color_mode='rgb', batch_size=32, image_size=(128, 256), shuffle=True, seed=123, validation_split=None, subset=None, interpolation='bilinear', follow_links=False)

data_iter = data.as_numpy_iterator()
batch = data_iter.next()







