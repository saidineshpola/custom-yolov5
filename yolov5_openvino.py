import numpy as np
import torch
import tensorflow as tf
import time
from ultralytics import YOLO
import numpy as np
from collections import Counter
import cv2
from openvino.inference_engine import IECore
from my_models.yolov5_model import YOLOModel as YOLOv5Model
from PIL import Image

int_to_class = {
    0: 'aegypti',
    1: 'albopictus',
    2: 'anopheles',
    3: 'culex',
    4: 'culiseta',
    5: 'japonicus-koreicus'
}
IMAGE_SIZE = [512, 512]


def prepare_image(image, target_size=(512, 512), target_layout="NHWC"):
    image = tf.image.resize_with_pad(image, target_height=IMAGE_SIZE[0], target_width=IMAGE_SIZE[1]) / 255.0

    # Normalize the image using given mean and std
    image = tf.clip_by_value(image, 0.0, 1.0)  # Ensure pixel values are in [0, 1]
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std  # Normalize image
    return image


# if it is running on GPU set device to cuda else cpu
device = 'CPU'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def classify_image(model, image, imgsz=512):
    result = model(image, imgsz=imgsz, device=device)  # conf=0.1,
    bboxes = result[0].boxes
    pred_class = ""
    result_l = []
    conf = 0
    if len(bboxes):
        conf, pred_class, result_l = bboxes[0].conf.cpu().numpy(
        ), model.names[int(bboxes[0].cls)], bboxes[0].xyxy.cpu().numpy()
    else:
        print('\033[31m No results from yolov8 model! \033[0m')

    return conf, pred_class, result_l

# getting mosquito_class name from predicted result


def extract_predicted_mosquito_class_name(extractedInformation):
    mosquito_class = ""
    if extractedInformation is not None:
        mosquito_class = str(extractedInformation.get("name").get(0))
    return mosquito_class

# getting mosquito_class number from predicted result


def extract_predicted_mosquito_class_number(extractedInformation):
    mosquito_class = ""
    if extractedInformation is not None:
        mosquito_class = str(extractedInformation.get("class").get(0))
    return mosquito_class

# getting mosquito_class confidence score from predicted result


def extract_predicted_mosquito_class_confidence(extractedInformation):
    mosquito_class = ""
    if extractedInformation is not None:
        mosquito_class = str(extractedInformation.get("confidence").get(0))
    return mosquito_class


def extract_predicted_mosquito_bbox(extractedInformation):
    bbox = []
    if extractedInformation is not None:
        xmin = str(extractedInformation.get("xmin").get(0))
        ymin = str(extractedInformation.get("ymin").get(0))
        xmax = str(extractedInformation.get("xmax").get(0))
        ymax = str(extractedInformation.get("ymax").get(0))
        bbox = [xmin, ymin, xmax, ymax]
    return bbox


class YOLOModel:
    def __init__(self):
        model_path = 'my_models/yolo_model_weights/openvino-tf'
        # Initialize inference engine
        ie = IECore()
        network = ie.read_network(model_path + '/model.xml',
                                  model_path + '/model.bin')
        self.exec_network = ie.load_network(network=network, device_name=device, num_requests=1)

        # run model for 3 random images to remove first time loading delay
        imgsz = IMAGE_SIZE[0]
        self.input_name = next(iter(network.input_info))
        for i in range(3):
            image = np.random.randint(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)
            image_prepared = prepare_image(image)
            output = self.exec_network.infer({self.input_name: image_prepared})
            detections = output
            for key, value in detections.items():
                print(key, value.shape)
        print('\033[32m Model Loaded Successfully \033[0m')

    def predict(self, image):
        start_time = time.time()
        all_predicted_classes = []
        all_bboxes = []
        conf = -1
        image_prepared = prepare_image(image)
        output = self.exec_network.infer({self.input_name: image_prepared})
        print('output', output)

        avg_bbox = [0, 0, 0, 0]
        if most_common_class == 'japonicus/koreicus':
            most_common_class = 'japonicus-koreicus'

        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time} seconds")
        # print(most_common_class, avg_bbox)
        return most_common_class, avg_bbox


class YOLOModel:
    def __init__(self):
        # "my_models/yolo_model_weights/last_openvino_model"
        self.yolo_model = YOLOv5Model()
        self.add_margin = False
        model_path = 'my_models/yolo_model_weights/openvino-tf'
        # Initialize inference engine
        ie = IECore()
        network = ie.read_network(model_path + '/model.xml',
                                  model_path + '/model.bin')
        self.exec_network = ie.load_network(network=network, device_name=device, num_requests=1)
        self.input_name = next(iter(network.input_info))
        imgsz = 512
        for i in range(3):
            image = np.random.randint(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)
            image_prepared = prepare_image(image)
            output = self.exec_network.infer({self.input_name: image_prepared})
        print('\033[32m Model Loaded Successfully \033[0m')

    def predict(self, image: np.ndarray,):
        start_time = time.time()
        mosquito_class_name_predicted = ""
        mosquito_class_bbox = [0, 0, 0, 0]
        det_class, det_bbox = self.yolo_model.predict(image)
        if det_bbox != mosquito_class_bbox:

            height, width, _ = image.shape
            # if the area of the bbox is less than 225*225 then we will not crop
            if self.add_margin and (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]) < 225 * 225:
                margin = max(det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1]) * 0.1
                crop_x1 = max(0, int(det_bbox[0] - margin))
                crop_y1 = max(0, int(det_bbox[1] - margin))
                crop_x2 = min(width, int(det_bbox[2] + margin))
                crop_y2 = min(height, int(det_bbox[3] + margin))
                image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            else:
                image = image[det_bbox[1]:det_bbox[3], det_bbox[0]:det_bbox[2]]

            image_prepared = prepare_image(image)
            output = self.exec_network.infer({self.input_name: image_prepared})
            mosquito_class_name_predicted = int_to_class[np.argmax(output['maxvit_small/predictions/Softmax'])]
            print('Predicted class: ', mosquito_class_name_predicted)
            mosquito_class_bbox = det_bbox
        bbox = [int(float(mcb)) for mcb in det_bbox]
        elapsed_time = time.time() - start_time
        print(f"\033[32m CLS Time taken: {elapsed_time} seconds \033[0m")
        return mosquito_class_name_predicted, bbox
