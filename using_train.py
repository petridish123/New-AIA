om PIL import Image
import os
import matplotlib.pyplot as plt
import statistics
import numpy as np

filepath = "bev_classification/images/train" # creating the base filepath
# Train has 0 - 69

#creating variables for comparison when finding the max and minimum height and width of the images
width = 200
height = 200
max_width = 0
max_height = 0
# Tuples are for the size vecotor for the minimum image sizes
tuple_width = ()
tuple_height = ()
# This is for the files that we decided to remove
width_file = ""
height_file = ""
# Vectors for the scatter plots
x_scatter = []
y_scatter = []
# for i in range(0,70): # iterating through all the training folders
#     filen = filepath + "_" +str(i) # creating a filepath for a basis
#     directory = os.fsencode(filen) # finds the directory name
#     print(directory)
directory = os.fsencode(filepath + "train_0")
for file in os.listdir(directory): # iterating through each folder in each training folder
    filename = os.fsdecode(file)
    path = filen + "/" + str(filename) # creating a base path
    if filename.endswith(".csv"): # not reading the .csv file
        continue
    for image in os.listdir(path): # reading each image
        ima = os.fsdecode(image) 
        if not ima.endswith(".jpg"):
            continue   
        if ima.endswith(".jpg"):

            file = os.fsdecode(image)
            whole_path = str(path) + "/" +str(file)
            im = Image.open(str(path) + "/" +str(file))

        x_scatter.append(im.size[0])
        y_scatter.append(im.size[1])
        im_width, im_height = im.size
