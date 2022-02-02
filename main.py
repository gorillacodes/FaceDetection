import cv2
import os
import numpy as np

os.chdir("C:/Users/Ahmar Ali Khan/Downloads/Talha/Tensor/OpenCV/Face and Eye Recognition")

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,h,w) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 5)
        gray_mask = gray[y:y+h, x:x+w]
        color_mask = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray_mask, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_mask, (ex, ey), (ex+ew, ey+eh), (0,255,0), 5)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



