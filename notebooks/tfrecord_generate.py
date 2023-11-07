import argparse
import tensorflow as tf
import os
import random

# Function to create a TFRecord example
# Define a mapping of class names to class IDs
class_to_id = {
    'aegypti': 0,
    'albopictus': 1,
    'anopheles': 2,
    'culex': 3,
    'culiseta': 4,
    'japonicus-koreicus': 5,
}


def create_tfrecord_example(image_path, class_name, target_size=(640, 640)):
    # Read the image file
    image = tf.io.read_file(image_path)
    # Decode the image (assuming JPEG format)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize_with_pad(image, target_height=640,
                                     target_width=640)  # image, target_size)
    image = tf.cast(image, tf.uint8)  # Convert to uint8
    encoded_image = tf.image.encode_jpeg(image).numpy()  # Encode as JPEG

    # Get the class ID from the class_name
    label = class_to_id[class_name]

    # Create a feature dictionary
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }

    # Create an Example object
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example

# Function to create a TFRecord file


def create_tfrecord_file(data_dir, tfrecord_filename):
    # Get a list of all class directories
    class_dirs = os.listdir(data_dir)

    # Create a TFRecord writer
    with tf.io.TFRecordWriter(tfrecord_filename) as writer:
        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)

            # Get a list of all image files in the class directory
            image_files = [os.path.join(class_path, filename)
                           for filename in os.listdir(class_path) if filename.endswith('.jpeg')]

            # Shuffle the image files to mix the data
            random.shuffle(image_files)

            # Loop through the image files and create TFRecord examples
            for image_path in image_files:
                example = create_tfrecord_example(image_path, class_dir)
                writer.write(example.SerializeToString())


def main(args):
    # Define paths to your train and val directories
    train_data_dir = args.data_dir + '/train'
    val_data_dir = args.data_dir + '/val'

    # Define the output paths for TFRecord files
    train_tfrecord_filename = args.train_tfrecord_filename or '../train_crop20f10.tfrecords'
    val_tfrecord_filename = args.val_tfrecord_filename or '../val_crop20f10.tfrecords'

    # Create TFRecord files for train and val
    create_tfrecord_file(train_data_dir, train_tfrecord_filename)
    create_tfrecord_file(val_data_dir, val_tfrecord_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create TFRecord files for training and validation datasets.')
    parser.add_argument('--data_dir',
                        default='/home/saidinesh/Desktop/Projects/yolov5/datasets/crop-datasets/ensemble1/20fold/fold_10',
                        help='Path to the training data directory')
    parser.add_argument('--train_tfrecord_filename', default=None,
                        help='Filename for the training TFRecord file (default: train_crop.tfrecords)')
    parser.add_argument('--val_tfrecord_filename', default=None,
                        help='Filename for the validation TFRecord file (default: val_crop.tfrecords)')

    args = parser.parse_args()
    main(args)
