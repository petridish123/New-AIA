import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import pandas as pd

# Load image dataset
data = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/julia/Desktop/New-AIA/bev_classification/condensed',
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(128, 256),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training',  # Use training subset for training
    interpolation='bilinear',
    follow_links=False
)

val_data = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/julia/Desktop/New-AIA/bev_classification/condensed',
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(128, 256),
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
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(8,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(99, activation='softmax'))  # Set output layer to number of classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0015), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
logdir = "logs"
tensorboard_callback = TensorBoard(log_dir=logdir)
history = model.fit(data, validation_data=val_data, epochs=12, callbacks=[tensorboard_callback])

# Save the model weights
model.save_weights('final_model_weights.h5')

# Extract loss and accuracy metrics
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Create a dataframe to store metrics
metrics_df = pd.DataFrame({'train_loss': train_loss,
                           'val_loss': val_loss,
                           'train_acc': train_acc,
                           'val_acc': val_acc})

# Save dataframe to CSV
metrics_df.to_csv('metrics5.csv', index=False)
# Metrics: Original Model
# Metrics2: Modify the complexities of the filters
# Metrics3: Resize the images to 128 x 256
# Metrics4: +1 Layer and increased learning