import os

# List of classes to be merged
all_classes = ['aegypti', 'albopictus', 'anopheles', 'culex', 'culiseta', 'japonicus-koreicus']
remaining_class = "remaining-class"

# Path to the directory containing label files
label_dir_train = "/home/saidinesh/Desktop/Projects/yolov5/datasets/new-det/train/labels/"
label_dir_val = "/home/saidinesh/Desktop/Projects/yolov5/datasets/new-det/val/labels/"

# Function to process label files and merge classes


def merge_classes(label_dir):
    label_files = os.listdir(label_dir)

    for file in label_files:
        with open(os.path.join(label_dir, file), "r") as f:
            lines = f.readlines()

        with open(os.path.join(label_dir, file), "w") as f:
            for line in lines:
                class_id, x_center, y_center, width, height = line.strip().split()
                class_name = all_classes[int(class_id)]
                new_class_id = "0" if class_name == "japonicus-koreicus" else "1"
                f.write(f"{new_class_id} {x_center} {y_center} {width} {height}\n")


# Merge classes for training data
merge_classes(label_dir_train)

# Merge classes for validation data
merge_classes(label_dir_val)
