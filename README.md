# Real-time Facial Recognition using YOLO and FaceNet

## Description
We implemented a small real-time facial recognition system, it will use camera to take pictures and render a real-time video to tell if the people in front of the camera is someone in our database (with his/her name as label) or someone unknown. The main algorithms we used are **YOLO v3** (You Only Look Once) and **FaceNet**.
* YOLO v3 is a state-of-the-art, real-time object detection algorithm. The published model recognizes 80 different objects in images and videos. However, we only use YOLO to detect faces in our project. For more details about YOLO v3, you check this [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).
* FaceNet develops a deep convolutional network to learn a mapping from face images to a compact Euclidean space where distances
directly correspond to a measure of face similarity, and it uses **triplet-loss** as its loss function. If you want to know more about the details of [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) and [triplet-loss](https://omoindrot.github.io/triplet-loss).

By the way, this is a final project of course EECS-496 (Advanced Deep Learning) at Northwestern University.

## Available Funtions
* **Face Alignment:** We have two versions of algorithms to detect and crop the faces in a picture — MTCNN and YOLO v3.
* **Training on FaceNet:** You can either train your model from scratch or use a pre-trained model for transfer learning. The loss function we use is triplet-loss.
* **Real-time Facial Recognition:** We use opencv to renver a real-time video after facial recognition and labeling.



## Usage

For Windows
```bash
$ python realitme_facenet.py
```

## Inspiration

* davidsandberg https://github.com/davidsandberg/facenet

  Provided the weights of FaceNet Model 20170512-110547, which were used as the starting point for training our FaceNet model


* sthanhng https://github.com/sthanhng/yoloface

  Provided a YOLO model trained on WIDER FACE for real-time facial detection


* cryer https://github.com/cryer/face_recognition

  Provided a framework for moving images from webcam to model, model to real-time on-screen bounding boxes and names
