import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# Load image dataset
data = tf.keras.utils.image_dataset_from_directory(
    '/Users/jedwoods/personal_projects/new_delicious/New-AIA/bev_classification/images/train_0',
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training',  # Use training subset for training
    interpolation='bilinear',
    follow_links=False
)

val_data = tf.keras.utils.image_dataset_from_directory(
    '/Users/jedwoods/personal_projects/new_delicious/New-AIA/bev_classification/images/train_0',
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation',  # Use validation subset for validation
    interpolation='bilinear',
    follow_links=False
)

# Normalize pixel values
data = data.map(lambda x, y: (x/255, y))
val_data = val_data.map(lambda x, y: (x/255, y))

# CNN model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(99, activation='softmax'))  # Set output layer to number of classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
logdir = "logs"
tensorboard_callback = TensorBoard(log_dir=logdir)
history = model.fit(data, validation_data=val_data, epochs=20, callbacks=[tensorboard_callback])
