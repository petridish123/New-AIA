from PIL import Image
import os
import matplotlib.pyplot as plt
import statistics
import numpy as np
import pandas as p


filepath = "bev_classification/images/train" # creating the base filepath

out_file = open("combined.csv", "w")

for i in range(0,70): # iterating through all the training folders
    filen = filepath + "_" +str(i) # creating a filepath for a basis
    directory = os.fsencode(filen) # finds the directory name
    print(directory)
    for file in os.listdir(directory): # iterating through each folder in each training folder
        filename = os.fsdecode(file)
        path = filen + "/" + str(filename) # creating a base path
        if not filename.endswith(".csv"): # guarding for only .csv
            continue
        cur_file = open(path, "r")
        for line in cur_file:
            out_file.write(line)
        cur_file.close()
        


out_file.close()