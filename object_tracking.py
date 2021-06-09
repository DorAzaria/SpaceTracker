import cv2
from day_detection import DayMode
from night_detection import NightMode
import numpy as np


class ObjectTracking:

    def __init__(self):
        self.first_frame = True
        self.mode_flag = None
        self.mode = None
        self.position = None
        self.frame = None

    def track(self, fr, state) -> tuple:
        self.frame = fr

        if self.first_frame:
            # If return True, it is night mode, else it is day mode.
            self.mode_flag = self.nightModeCheck()
            self.first_frame = False

        # If it is night mode
        if self.mode_flag:
            self.position = self.mode.nightAction(self.frame, state)
        else:
            self.position = self.mode.dayAction(self.frame, state)

        self.GUI()

        return self.position

    def nightModeCheck(self) -> bool:
        lower_night = np.array([0, 0, 0])
        upper_night = np.array([179, 255, 155])
        mask = cv2.inRange(self.frame, lower_night, upper_night)
        mean = mask.mean()
        if mean > 250:
            self.mode = NightMode()
            return True
        self.mode = DayMode()
        return False

    def getFrame(self):
        return self.frame

    def GUI(self):
        centerX = self.position[0]
        centerY = self.position[1]

        if centerX != -1 and centerY != -1:

            (hc, wc) = self.frame.shape[:2]  # w:image-width and h:image-height
            fX = wc // 2
            fY = hc // 2
            cv2.circle(self.frame, (fX, fY), 2, (0, 0, 255), 4)
            X = int(centerX + (fX - centerX) * 0.5)
            Y = int(centerY + (fY - centerY) * 0.5)
            center = (X, Y)

            box = self.mode.getBox()
            self.drawBox(box)
            limit = self.checkArrowBound((box[0], box[1]), (box[2], box[3]), (fX, fY))

            if not limit:
                cv2.arrowedLine(self.frame, (fX, fY), center, (0, 0, 255), thickness=2, tipLength=0.2)

    @staticmethod
    def checkArrowBound(bl, tr, p):
        if bl[0] < p[0] < tr[0] and bl[1] < p[1] < tr[1]:
            return True
        else:
            return False

    def drawBox(self, box):
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(self.frame, (x, y), (w, h), (0, 255, 0), 3, 3)
        cv2.putText(self.frame, "ON TARGET!", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
