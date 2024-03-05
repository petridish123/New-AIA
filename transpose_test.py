from PIL import Image
import os
import matplotlib.pyplot as plt
import statistics
import numpy as np

filepath = "bev_classification/images/test" # creating the base filepath
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
for i in range(0,15): # iterating through all the training folders
    filen = filepath + "_" +str(i) # creating a filepath for a basis
    directory = os.fsencode(filen) # finds the directory name
    print(directory)
    for file in os.listdir(directory): # iterating through each folder in each training folder
        filename = os.fsdecode(file)
        path = filen + "/" + str(filename) # creating a base path
        if filename.endswith(".csv") or filename.endswith(".DS_Store"): # not reading the .csv file
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


                
            # checking all the image sizes
            if im.size[0] < width:
                width = im.size[0]
                tuple_width = im.size
                width_file = str(whole_path)
                
            if im.size[1] < height:
                height = im.size[1]
                tuple_height = im.size
                height_file = str(whole_path)


            if im.size[1] > max_height:
                max_height = im.size[1]
            if im.size[0] > max_width:
                max_width = im.size[0]
            
            # if the image size is too small, remove it from the training set, it would be useless.
 


# Printing the statistics we want
print(f"Width: {width}")
print(f"Height: {height}")
print(f"width : {tuple_width}, height: {tuple_height}")
print(f"height file: {height_file}")
print(f"width file: {width_file}")
print(f"max width: {max_width}, max heigh: {max_height}")

print(f"average height: {sum(y_scatter)/len(y_scatter)}")
print(f"average width: {sum(x_scatter)/len(x_scatter)}")

print(f"median width: {statistics.median(x_scatter)}")
print(f"median height: {statistics.median(y_scatter)}")

plt.scatter(x_scatter, y_scatter)
plt.xlabel("Width")
plt.ylabel("Height")
plt.savefig("Plot_test.png")
m,b = np.polyfit(x_scatter,y_scatter, 1)
x = np.array(x_scatter)
plt.plot(x, m*x+ b, color = "red")
plt.savefig("Test_regression_pre_T.png")
plt.clf()


m,b = np.polyfit(x_scatter,y_scatter, 1)
x = np.array(x_scatter)
plt.plot(x, m*x+ b, color = "red")


trans_x = []
trans_y = []
for i in range(0,15): # iterating through all the training folders
    filen = filepath + "_" +str(i) # creating a filepath for a basis
    directory = os.fsencode(filen) # finds the directory name
    print(directory)
    for file in os.listdir(directory): # iterating through each folder in each training folder
        filename = os.fsdecode(file)
        path = filen + "/" + str(filename) # creating a base path
        if filename.endswith(".csv") or filename.endswith(".DS_Store"): # not reading the .csv file
            continue
        for image in os.listdir(path): # reading each image
            ima = os.fsdecode(image) 
            if not ima.endswith(".jpg"):
                continue   
            if ima.endswith(".jpg"):
                file = os.fsdecode(image)
                whole_path = str(path) + "/" +str(file)
                im = Image.open(str(path) + "/" +str(file))


                im_width, im_height = im.size


                if im_width > im_height:
                    transposed_image = im.transpose(Image.TRANSPOSE)
                    transposed_image.save(str(path) + "/" + str(file))

                    t_width, t_height = transposed_image.size

                    trans_x.append(t_width)
                    trans_y.append(t_height)
"""
            if im_width > im_height:
                transposed_image = im.transpose(Image.TRANSPOSE)
                transposed_image.save(str(path) + "/" + str(file))

"""

plt.scatter(trans_x,trans_y)
# plt.plot(x, m*x+b, color = "red")
plt.savefig("transposed_test.png")
plt.show()