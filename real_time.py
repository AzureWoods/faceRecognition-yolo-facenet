from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
#import detect_face
import os
import time
import pickle

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        HumanNames = ['liuzheng','shixing','xuguanyu','Human_h']    #train human name

        print('Loading feature extraction model')
        modeldir = './models/'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = './myclassifier/my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        c = 0

        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()
            classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faceRects = classfier.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

                if len(faceRects) > 0:
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []

                    emb_array = np.zeros((1, embedding_size))
                    for faceRect in faceRects:
                        x, y, w, h = faceRect
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)

                        cropped.append(frame[y - 10:y + h + 10,x - 10:x + w + 10, :])
                        cropped[0] = facenet.flip(cropped[0], False)
                        scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                        scaled[0] = cv2.resize(scaled[0], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[0] = facenet.prewhiten(scaled[0])
                        scaled_reshape.append(scaled[0].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        for H_i in HumanNames:
                            if HumanNames[best_class_indices[0]] == H_i:
                                result_names = HumanNames[best_class_indices[0]]
                                cv2.putText(frame, result_names, (x + 30, y - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
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
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
