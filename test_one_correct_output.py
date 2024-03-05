import os, json
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


def evaluate_images(model, test_file_path, output_file):
    with open(output_file, 'w') as f:
        with open(test_file_path, 'r') as test_file:
            with open('conversion.json', 'r') as json_file:
                loaded_dict = json.load(json_file)
                successes = 0
                lines = 0

                for line in test_file:
                    img_path = "C:/Users/julia/Desktop/New-AIA/bev_classification/" + line.strip()  # Update image path
                    img = image.load_img(img_path, target_size=(128, 256))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
                    img_array /= 255.0  # Normalize pixel values

                    prediction = model.predict(img_array)
                    predicted_class = np.argmax(prediction)

                    # # Determine if prediction is correct
                    # if line[13] == "/":
                    #     expected_class = int(line[14:26])
                    # else:
                    #     expected_class = int(line[15:27])

                    predicted_class = loaded_dict[str(predicted_class)]

                    # print(predicted_class, expected_class)
                    # is_correct = 1 if predicted_class == expected_class else 0
                    # successes += is_correct

                    f.write(f"{line.strip()},{predicted_class}\n")

                    

                # score = successes / lines
                # f.write(f"Final Score: {score}")
# Load the trained model
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
model.compile(optimizer=Adam(learning_rate=0.0012), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights('final_model_weights.h5')

# Path to the file containing image paths and labels
test_file_path = 'C:/Users/julia/Desktop/New-AIA/bev_classification/datasets/test_edited.txt'

# Output file path
output_file = 'test_one_formatted.txt'

# Evaluate images and save results
evaluate_images(model, test_file_path, output_file)
