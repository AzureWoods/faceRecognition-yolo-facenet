# Real-time Facial Recognition using YOLO and FaceNet

## Description
We implemented a small real-time facial recognition system, it will use camera to take pictures and render a real-time video to tell if the people in front of the camera is someone in our database (with his/her name as label) or someone unknown. The main algorithms we used are **YOLO v3** (You Only Look Once) and **FaceNet**.
* YOLO v3 is a state-of-the-art, real-time object detection algorithm. The published model recognizes 80 different objects in images and videos. However, we only use YOLO to detect faces in our project. For more details about YOLO v3, you check this [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).


This is a final project of course EECS-496 (Advanced Deep Learning) at Northwestern University.


## Usage

For Windows
```bash
$ python realitme_facenet.py
```

## Credit

* davidsandberg https://github.com/davidsandberg/facenet

  Provided the weights of FaceNet Model 20170512-110547, which were used as the starting point for training our FaceNet model


* sthanhng https://github.com/sthanhng/yoloface

  Provided a YOLO model trained on WIDER FACE for real-time facial detection


* cryer https://github.com/cryer/face_recognition

  Provided a framework for moving images from webcam to model, model to real-time on-screen bounding boxes and names
