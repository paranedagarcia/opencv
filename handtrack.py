# hand tracking
import cv2
import time
import mediapipe as mp
import time
import imutils

cap = cv2.VideoCapture(3) # 1: OBS, 3:poly
# write video
''' wrv = cv2.VideoWriter('data/hands.mp4', # file
    cv2.VideoWriter_fourcc(*'XVID'), # codec
    24.0, # frames
    (960, 540) # medidas
    ) '''

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    res, img = cap.read()
    if res == False: break

    img = imutils.resize(img, width=960)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    response = hands.process(imgRGB)

    if response.multi_hand_landmarks:
        for handland in response.multi_hand_landmarks: # each hand
            # id & position each landmark
            for id, lm in enumerate(handland.landmark):
                #print(id, lm)
                h, w, c = img.shape
                # center position detect
                cx, cy = int(lm.x*w), int(lm.y*h )
 
                if id == 0:
                    cv2.circle(img, (cx, cy),10, (0,250,250), cv2.FILLED)

            mpDraw.draw_landmarks(img, handland, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand", img)
    # wrv.write(img) # write video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# wrv.release()
cv2.destroyAllWindows()
