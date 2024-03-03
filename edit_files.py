import os
import shutil


train_data = ""

dir = "/Users/jedwoods/personal_projects/new_delicious/New-AIA/bev_classification/train"
orig = "/Users/jedwoods/personal_projects/new_delicious/New-AIA/bev_classification"
if not os.path.exists(dir):
    os.makedirs(dir)

with open('/Users/jedwoods/personal_projects/new_delicious/New-AIA/bev_classification/datasets/train.txt', 'r') as f:

    for line in f:
        line = line.strip()
        path = line.split(',')[0]
        if "train" in path:
            print(path)
            train_data = path.split("/")[3]
            print(train_data)
            class_dir = orig+ "/" + line.split(",")[1].strip()
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            shutil.copy(orig + + "/" + path, "/Users/jedwoods/personal_projects/new_delicious/New-AIA/bev_classification/images/train/" + train_data)
         
      