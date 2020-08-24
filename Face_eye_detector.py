#Author: Emmanuel Efewongbe
#Details: Detects human face and eyes via the webcam using openCV. Press 'q' to quit.

import cv2

#import cascades
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

#capture video from webcam and set window size
img= cv2.VideoCapture(0)
img.set(3,1250)
img.set(4,750)

#while webcam still capturing
while True:
    #read images from webcam
    success, image = img.read()
    
    #convert each image to grey
    grey_conv = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    face_detect = face_cascade.detectMultiScale(image = grey_conv, scaleFactor = 1.1, minNeighbors = 5)
    print ('faces found: ', len(face_detect))

    #draw a red box around each face found and a green box around each eye found 
    for(x,y,w,h) in face_detect:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        eye_detect = eye_cascade.detectMultiScale(grey_conv[y:y+h, x:x+w])
        for(ex, ey, ew, eh) in eye_detect:
            cv2.rectangle(image[y:y+h, x:x+w], (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

    #show result
    cv2.imshow('img', image)
    
    #quit program when user types 'q'
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    
img.release()
cv2.destroyAllWindows()
