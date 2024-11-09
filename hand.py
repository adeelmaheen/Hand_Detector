import cv2
import mediapipe as mp

class FindHands():
    def __init__(self, detection_con=0.5, tracking_con=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=detection_con, min_tracking_confidence=tracking_con)
        self.mpDraw = mp.solutions.drawing_utils

    def getPosition(self, img, indexes, hand_no=0, draw=True):
        lst = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) >= hand_no + 1:
                for id, lm in enumerate(results.multi_hand_landmarks[hand_no].landmark):
                    if id in indexes:
                        h, w, c = img.shape
                        x, y = int(lm.x * w), int(lm.y * h)
                        lst.append((x, y))
                if draw:
                    self.mpDraw.draw_landmarks(img, results.multi_hand_landmarks[hand_no], self.mpHands.HAND_CONNECTIONS)
        return lst

    def finger_up(self, img, point1, point2, hand_no=0):
        pos = self.getPosition(img, (point1, point2), draw=False)
        if len(pos) == 2:
            return pos[0][1] < pos[1][1]  # Return True if the tip is above the lower joint
        else:
            return False

    def index_finger_up(self, img, hand_no=0):
        return self.finger_up(img, 6, 8, hand_no)

    def middle_finger_up(self, img, hand_no=0):
        return self.finger_up(img, 10, 12, hand_no)

    def ring_finger_up(self, img, hand_no=0):
        return self.finger_up(img, 14, 16, hand_no)

    def little_finger_up(self, img, hand_no=0):
        return self.finger_up(img, 18, 20, hand_no)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    hands = FindHands(detection_con=0.79)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Detect the hand landmarks
        lmlist = hands.getPosition(img, list(range(21)), draw=True)

        # Determine if fingers are up
        if len(lmlist) > 0:
            finger_status = []

            # Determine left or right hand
            if lmlist[5][0] > lmlist[17][0]:
                handType = "Right"
            else:
                handType = "Left"

            # Thumb logic
            if handType == "Right":
                finger_status.append(1 if lmlist[4][0] > lmlist[3][0] else 0)
            else:
                finger_status.append(1 if lmlist[4][0] < lmlist[3][0] else 0)

            # Other fingers logic
            finger_status.append(1 if hands.index_finger_up(img) else 0)
            finger_status.append(1 if hands.middle_finger_up(img) else 0)
            finger_status.append(1 if hands.ring_finger_up(img) else 0)
            finger_status.append(1 if hands.little_finger_up(img) else 0)

            total_fingers = finger_status.count(1)

            # Display the result on the screen
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        cv2.imshow("Image", img)
        if cv2.waitKey(10) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

ignore_missing_imports = True
