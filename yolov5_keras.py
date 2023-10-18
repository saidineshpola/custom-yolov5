import os
import numpy as np
import torch
import tensorflow as tf
import time
from ultralytics import YOLO
import numpy as np
from my_models.yolov5_model import YOLOModel as YOLOv5Model
from PIL import Image

np.random.seed(42)
tf.random.set_seed(42)

int_to_class = {
    0: 'aegypti',
    1: 'albopictus',
    2: 'anopheles',
    3: 'culex',
    4: 'culiseta',
    5: 'japonicus-koreicus'
}
IMAGE_SIZE = [512, 512]


def get_model(use_custom_layers=True):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
        input_shape=(512, 512, 3),
        include_top=False,
        weights='imagenet',
        pooling='max',
        # classes=6,
    )
    if use_custom_layers:
        x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)
        x = tf.keras.layers.Dropout(0.45)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.45)(x)
        outputs = tf.keras.layers.Dense(6, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    return model


def prepare_image(image, target_size=(512, 512)):
    # image = tf.keras.preprocessing.image.img_to_array(image)

    image = tf.cast(image, tf.float32)
    image = image  # / 255.0

    image = tf.image.resize(image, target_size)

    image = tf.expand_dims(image, 0)   # Add batch dimension

    return image


# if it is running on GPU set device to cuda else cpu
device = 'CPU'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set CPU in tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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


class YOLOModel:
    def __init__(self):
        # "my_models/yolo_model_weights/last_openvino_model"
        self.yolo_model = YOLOv5Model()
        self.add_margin = False
        # self.mm = maxvit.MaxViT_Small(
        #     input_shape=(512, 512, 3),
        #     num_classes=6,
        # )
        self.model = get_model()
        self.model.load_weights('my_models/yolo_model_weights/openvino-tf/efficientNetv2M-basic-best.h5')

        imgsz = 512
        for i in range(3):
            image = np.random.randint(0, 255, size=(imgsz, imgsz, 3), dtype=np.uint8)
            image_prepared = prepare_image(image)
            output = self.model.predict(image_prepared)
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
            output = self.model.predict(image_prepared)
            output = tf.nn.softmax(output[0])
            print(output)
            mosquito_class_name_predicted = int_to_class[np.argmax(output)]
            print('Predicted class: ', mosquito_class_name_predicted)
            mosquito_class_bbox = det_bbox
        bbox = [int(float(mcb)) for mcb in det_bbox]
        elapsed_time = time.time() - start_time
        print(f"\033[32m CLS Time taken: {elapsed_time} seconds \033[0m")
        return mosquito_class_name_predicted, bbox
