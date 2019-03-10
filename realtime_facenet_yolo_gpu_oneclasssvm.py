from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from yolo.yolo import *

from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo_weights/YOLO_Face.h5',
                    help='path to model weights file')
    parser.add_argument('--anchors', type=str, default='yolo_cfg/yolo_anchors.txt',
                    help='path to anchor definitions')
    parser.add_argument('--classes', type=str, default='yolo_cfg/face_classes.txt',
                    help='path to class definitions')
    parser.add_argument('--score', type=float, default=0.5,
                    help='the score threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                    help='the iou threshold')
    parser.add_argument('--img-size', type=list, action='store',
                    default=(416, 416), help='input image size')
    args = parser.parse_args()
    return args;

# print('Creating networks and loading parameters')

# Load YOLO V2 model.
# net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def _main():

    args = get_args()
    

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # pnet, rnet, onet = detect_face.create_mtcnn(sess, './models/')

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160

            print('Loading feature extraction model')
            modeldir = './models/facenet/20190310-055158'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            path = "./myclassifier/oneClassSVM/"
            path_exp = os.path.expanduser(path)
            paths = [path for path in os.listdir(path_exp) if os.path.isfile(os.path.join(path_exp, path))]
            print(paths)
            model_list = []
            for myModel in paths:
                classifier_filename = path + myModel
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, className) = pickle.load(infile)
                model_list.append([className, model])
                print('load classifier file-> %s' % classifier_filename_exp)

            video_capture = cv2.VideoCapture(0)
            c = 0

            print('Start Recognition!')
            prevTime = 0
            myYolo = YOLO(args)
            while True:
                ret, frame = video_capture.read()

                # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

                curTime = time.time()    # calc fps
                timeF = frame_interval

                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    #print(frame.shape[0])
                    #print(frame.shape[1])
                    
                    image = Image.fromarray(frame)
                    img, bounding_boxes = myYolo.detect_image(image)

                    # Remove the bounding boxes with low confidence
                    nrof_faces = len(bounding_boxes)
                    ## Use MTCNN to get the bounding boxes
                    # bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    # nrof_faces = bounding_boxes.shape[0]
                    #print('Detected_FaceNum: %d' % nrof_faces)

                    if nrof_faces > 0:
                        # det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        # cropped = []
                        # scaled = []
                        # scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = bounding_boxes[i][0]
                            bb[i][1] = bounding_boxes[i][1]
                            bb[i][2] = bounding_boxes[i][2]
                            bb[i][3] = bounding_boxes[i][3]

                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('face is inner of range!')
                                continue

                            # cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            # cropped[0] = facenet.flip(cropped[0], False)
                            # scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                            # scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
                            #                        interpolation=cv2.INTER_CUBIC)
                            # scaled[0] = facenet.prewhiten(scaled[0])
                            # scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                            # feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}

                            cropped = (frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            print("{0} {1} {2} {3}".format(bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
                            cropped = facenet.flip(cropped, False)
                            scaled = (misc.imresize(cropped, (image_size, image_size), interp='bilinear'))
                            scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = (scaled.reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}

                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                            label = None
                            for myModel in model_list:
                                myClass = myModel[0]
                                svmModel = myModel[1]
                                predictions = svmModel.predict(emb_array)
                                print(predictions)
                                if predictions == 1:
                                    label = myClass
                                    break

                            # predictions = model.predict_proba(emb_array)
                            # best_class_indices = np.argmax(predictions, axis=1)
                            # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            # for H_i in HumanNames:
                            #     if HumanNames[best_class_indices[0]] == H_i:
                            result_names = "Unknown" if label == None else label
                            #print(result_names)
                            cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
                    else:
                        print('Unable to align')

                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                str = 'FPS: %2.3f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, str, (text_fps_x, text_fps_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                # c+=1
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            # #video writer
            # out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    _main()