import os
import shutil

'''NOTE: THIS CODE WILL MOVE ALL THE IMAGES OUT OF THE TRAINING DATASETS INTO A SINGLE FOLDER!'''

# Function to condense folders
def condense_folders(source_dir, destination_dir):
    # Loop through train folders
    for i in range(70):
        train_folder = os.path.join(source_dir, f"train_{i}")
        # Check if train folder exists
        if os.path.exists(train_folder):
            # Loop through subfolders in train folder
            for subdir in os.listdir(train_folder):
                subdir_path = os.path.join(train_folder, subdir)
                # Check if it's a directory
                if os.path.isdir(subdir_path):
                    # Create destination directory if it doesn't exist
                    dest_folder = os.path.join(destination_dir, subdir)
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                        print(f"Created folder {dest_folder}")
                    else:
                        pass
                        # print(f"Destination folder '{dest_folder}' already exists. Skipping...")
                    # Move files from subfolder to destination folder
                    for file_name in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file_name)
                        dest_file_path = os.path.join(dest_folder, file_name)
                        # Check if the file already exists in the destination folder
                        if not os.path.exists(dest_file_path):
                            shutil.move(file_path, dest_folder) # TODO: CHANGE THIS TO .copy IF YOU WANT TO COPY THE DATASET. WILL TAKE MUCH LONGER AND WILL REQUIRE YOU TO RUN WITH ADMIN PRIVILEGES
                        else:
                            print(f"File '{file_name}' already exists in destination folder '{dest_folder}'. Skipping...")

# Example usage
source_directory = "" # TODO: PUT THE PATH TO THE FOLDER CONTAINING ALL DATASETS
destination_directory = "" # TODO: SET AN OUTPUT PATH
condense_folders(source_directory, destination_directory)
