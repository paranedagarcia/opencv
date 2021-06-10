# detect finger movement
# 2021-06-10
import cv2
import mediapipe as mp
import imutils
from numpy.lib.function_base import append
import handDetect as hdt

wcam, hcam = 960, 540 # half fullHD

cap = cv2.VideoCapture(3)
detector = hdt.handDetect()

tipIds = [4, 8, 12, 16, 20]

while True:
    res, img = cap.read()
    img = imutils.resize(img, width=960)
    img = detector.findHand(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        fingers = []
        if lmList[tipIds[0]][1] > lmList[tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1,5):
            if lmList[tipIds[id]][2] > lmList[tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalFingers = fingers.count(1)
        print(totalFingers)

        #print(lmlist)

    cv2.imshow("hands", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()