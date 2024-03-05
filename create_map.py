import os
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import json

def create_optimized_dictionary(input_dict):
    optimized_dict = {}
    
    # Iterate over the input dictionary
    for key, sub_dict in input_dict.items():
        max_occurrences = 0
        max_number = None
        
        # Find the 12-digit number with the highest number of occurrences
        for number, occurrences in sub_dict.items():
            if occurrences > max_occurrences:
                max_occurrences = occurrences
                max_number = number
        
        # Update the optimized dictionary with the highest occurrences
        if max_number is not None:
            optimized_dict[int(key)] = max_number  # Convert key to int
            
    return optimized_dict

# Example dictionary
input_dict = {
    65: {610764022677: 1, 610764022868: 1, 70847012474: 1},
    13: {12000212024: 3},
    74: {611269113570: 2, 12000212062: 1},
    # More entries...
}

def evaluate_images(model, test_file_path, output_file):
    dictionary = {}
    with open(output_file, 'w') as f:
        with open(test_file_path, 'r') as test_file:
            i = 0
            for line in test_file:
                img_path = "C:/Users/julia/Desktop/New-AIA/bev_classification/" + line.strip()  # Update image path
                img = image.load_img(img_path, target_size=(128, 256))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
                img_array /= 255.0  # Normalize pixel values

                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)

                # Determine if prediction is correct
                if line[13] == "/":
                    expected_class = int(line[14:26])
                else:
                    expected_class = int(line[15:27])

                print(predicted_class, expected_class)

                if not predicted_class in dictionary:
                    dictionary[predicted_class] = {}

                if not expected_class in dictionary[predicted_class]:
                    dictionary[predicted_class][expected_class] = 0
                dictionary[predicted_class][expected_class] += 1
                    

                # is_correct = 1 if predicted_class == expected_class else 0

                # f.write(f"{img_path}: {is_correct}\n")
                i += 1

                if i >= 9999:
                    break

            optimized_dict = create_optimized_dictionary(dictionary)
            optimized_dict_sorted = dict(sorted(optimized_dict.items()))

            # Write the dictionary to a JSON file
            with open('conversion.json', 'w') as json_file:
                json.dump(optimized_dict_sorted, json_file)

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
output_file = 'test_results.txt'

# Evaluate images and save results
evaluate_images(model, test_file_path, output_file)
