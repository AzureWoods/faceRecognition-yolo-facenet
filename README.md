# face_recognition
real time face recognition with MTCNN and FaceNet

## Before run code

you need to do things below:

*  I have already uploaded det1.npy det2.npy det3.npy which for MTCNN,but you still need to download facenet's pb file from [davidsandberg's
github](https://github.com/davidsandberg/facenet) like 20170511-185253,extract to pb file and put in models directory.
* tensorflow-gpu 1.1.0 , later version may also work.
* python 3.X


## Inspiration

* OpenFace
* [davidsandberg's github](https://github.com/davidsandberg/facenet)
* main code is refered to bearsprogrammer

## Something note

`Remember to change some codes where you need to put your own name and your friends' name instead of mine.`

## Run code

Do as follows step by step:

* To make you easy to get your photo and put in right structure as I said in intput and output directorys' readme.md file,I 
already privide getphoto.py which can take photos by openCV and autoly put it in input directory as format.
* Next,you need to run Make_aligndata.py to align your photos which only croped your face part and autoly put in output directory as format.This photos will be used to train our own classifier.
* Run Make_classifier.py to train our own classifier with SVM.Of course you can use your own classifier if you want.Then you may 
see myclassifier.pkl file in myclassifier directory.
* Finally,run realtime_facenet.py or real_time.py. 
realtime_facenet.py is MTCNN version.real_time.py is another choice which use haar detector in openCV instead of MTCNN.

## Result

If everything is ok ,you will see result below:

![](https://github.com/cryer/face_recognition/raw/master/image/1.png)

## More

I used Chinese to do some Introduction about MTCNN and FaceNet.[See my blog for details](https://cryer.github.io/2018/01/facerecognition/)
