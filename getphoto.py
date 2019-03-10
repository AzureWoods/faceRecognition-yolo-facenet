import cv2

video_capture = cv2.VideoCapture(0)
c = 0
while (True):
    ret, frame = video_capture.read()
    classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")

    faceRects = classfier.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects) == 1:
        c += 1
        if c % 10 ==0:
            cv2.imwrite('input/xuguanyu/' + str(int(c/10)) + '.jpg', frame)


    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
