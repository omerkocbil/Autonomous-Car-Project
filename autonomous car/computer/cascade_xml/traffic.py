import numpy as np
import cv2 as cv
stop_sign_cascade = cv.CascadeClassifier('stop_sign.xml')
#eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
img = cv.imread('tabela.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

signboards = stop_sign_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in signboards:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
