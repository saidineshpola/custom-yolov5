import os
import shutil

# Define paths
split_dir = '../datasets/crop-datasets/ensemble1/kfold20/fold_19/val/'
train_dataset_dir = '../datasets/crop-datasets/ensemble1/classify-crop-all/train'
val_dataset_dir = '../datasets/crop-datasets/ensemble1/classify-crop-all/val'

# Get class names from the split directory
class_names = os.listdir(split_dir)

# Iterate through class names
for class_name in class_names:
    # Path to the training class directory
    original_image_class = os.path.join(split_dir, class_name)
    train_class_dir = os.path.join(train_dataset_dir, class_name)
    # Path to the validation class directory
    val_class_dir = os.path.join(val_dataset_dir, class_name)
    os.makedirs(val_class_dir, exist_ok=True)

    # List all files in the training class directory
    files = os.listdir(original_image_class)

    # Move files from training to validation
    for file_name in files:
        src_path = os.path.join(train_class_dir, file_name)
        dest_path = os.path.join(val_class_dir, file_name)
        shutil.move(src_path, dest_path)

print("Validation dataset created successfully.")
