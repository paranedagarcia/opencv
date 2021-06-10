# 2021-06-10
import cv2
import mediapipe as mp
import imutils

class handDetect():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.response = self.hands.process(imgRGB)

        if self.response.multi_hand_landmarks:
            for handland in self.response.multi_hand_landmarks: # para cada mano
                if draw:
                    self.mpDraw.draw_landmarks(img, handland, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.response.multi_hand_landmarks:
            mihand = self.response.multi_hand_landmarks[handNo]
            for id, lm in enumerate(mihand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h )
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy),8, (0,0,200), cv2.FILLED)
        return lmlist


def main():
    cap = cv2.VideoCapture(3)
    detector = handDetect()
    while True:
        res, img = cap.read()
        #img = imutils.resize(img, width=720)
        img = detector.findHand(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist)

        cv2.imshow("hands", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()