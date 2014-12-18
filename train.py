import numpy as np
import cv2

cascade = cv2.CascadeClassifier('samples2/cascade.xml')
img = cv2.imread('test/t2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

muscles = cascade.detectMultiScale(gray, 1.3, 5)
print muscles
for (x,y,w,h) in muscles:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
cv2.imshow('img',img)

sift = cv2.SIFT()
kp = sift.detect(gray,None)
print kp
img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
