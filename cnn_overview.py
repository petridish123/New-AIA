from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a Sequential model
model = Sequential()

# Add a convolutional layer with 32 filters, a kernel size of (3, 3), and relu activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add a max pooling layer with a pool size of (2, 2)
model.add(MaxPooling2D((2, 2)))

# Flatten the output of the previous layer
model.add(Flatten())

# Add a fully connected layer with 128 neurons and relu activation
model.add(Dense(128, activation='relu'))

# Add the output layer with 10 neurons (one for each class in MNIST) and softmax activation
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
