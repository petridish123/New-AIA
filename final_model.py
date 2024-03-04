import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import os
import pickle
import matplotlib.pyplot as plt

# Get the current working directory
cwd = os.getcwd()

# Construct the filepath for the training set
train_filepath = os.path.join(cwd, 'bev_classification', 'images', 'train_0')

# Load image dataset
data = tf.keras.utils.image_dataset_from_directory(
    train_filepath,
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
    train_filepath,
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


#graph the loss and accuracy of the model over time
# Graph the loss and accuracy of the model over time
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot validation loss
plt.subplot(2, 2, 2)
plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# Set y-axis range for both plots
plt.ylim([0.0, max(max(history.history['loss']), max(history.history['val_loss']))])


# Plot training accuracy
plt.subplot(2, 2, 3)
plt.plot(history.history['accuracy'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1.0])  # Set y-axis range from 0.0 to 1.0

# Plot validation accuracy
plt.subplot(2, 2, 4)
plt.plot(history.history['val_accuracy'])
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1.0])  # Set y-axis range from 0.0 to 1.0

plt.tight_layout()

# Save the figure to a folder called "train_v_validation"
save_folder = "train_v_validation"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "figure.png")
plt.savefig(save_path)

plt.show()


# Pickle the model weights and biases
model_weights = model.get_weights()
with open('model_weights.pkl', 'wb') as f:
    pickle.dump(model_weights, f)
    
