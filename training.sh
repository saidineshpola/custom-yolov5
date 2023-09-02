#!/bin/bash

# Command 1
python train.py --img 640 --epochs 100 --batch-size 8 --save-period 10 --data datasets/dataset.yaml --weights yolov5s.pt --name yolov5s-elu

# Command 2
python train.py --img 640 --epochs 100 --batch-size 4 --save-period 10 --data datasets/dataset.yaml --weights yolov5l.pt --name yolov5l-elu
