# detect finger movement
# 2021-06-10
import cv2
import mediapipe as mp
import imutils
import handDetect as hdt

wcam, hcam = 960, 540 # half fullHD

cap = cv2.VideoCapture(3)
detector = hdt.handDetect()
while True:
    res, img = cap.read()
    img = imutils.resize(img, width=960)
    img = detector.findHand(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        print(lmlist)

    cv2.imshow("hands", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()