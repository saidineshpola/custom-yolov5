import os
from sklearn.model_selection import KFold
import shutil
from tqdm import tqdm
# Define the number of folds
num_folds = 10

# Define the path to the dataset directory
data_dir = '/home/saidinesh/Desktop/Projects/yolov5/datasets/crop-datasets/ensemble1/train'

# Get the list of class folders
class_folders = os.listdir(data_dir)

# Initialize KFold cross-validator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate through the class folders
for class_folder in tqdm(class_folders):
    class_path = os.path.join(data_dir, class_folder)

    # Get a list of image files in the current class folder
    image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

    # Split the data into folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        # Create folders for the current fold if they don't exist
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx + 1}')
        os.makedirs(os.path.join(fold_dir, 'train', class_folder), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'val', class_folder), exist_ok=True)

        # Move files to appropriate fold folders
        for idx in train_idx:
            src_file = os.path.join(class_path, image_files[idx])
            dest_file = os.path.join(fold_dir, 'train', class_folder, image_files[idx])
            shutil.copy(src_file, dest_file)

        for idx in val_idx:
            src_file = os.path.join(class_path, image_files[idx])
            dest_file = os.path.join(fold_dir, 'val', class_folder, image_files[idx])
            shutil.copy(src_file, dest_file)

print("Data split into 10 folds with 90% training and 10% validation for each fold and each class.")
