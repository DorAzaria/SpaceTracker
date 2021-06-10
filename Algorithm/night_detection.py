import cv2 as cv
from numpy import long
import statistics


class NightMode:

    def __init__(self):
        self.stat = []
        self.avg_contours = []
        self.tempMaxLoc = (0, 0)
        self.backSub = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)
        self.backSub.setNMixtures(8)
        self.lastMean = 0
        self.tracker = cv.legacy_TrackerCSRT.create()
        self.target_flag = False
        self.frame = None
        self.bbox = None

    def nightAction(self, fr, state) -> tuple:
        self.frame = fr
        cv.putText(self.frame, "Sky Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        return self.skyMode(state)

    def skyMode(self, state) -> tuple:

        fgMask = self.backSub.apply(self.frame)

        fram, thresh = cv.threshold(fgMask, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        avg = len(contours)
        self.avg_contours.append(avg)

        gray = cv.GaussianBlur(fgMask, (7, 7), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
        locMean = (maxLoc[0] + maxLoc[1]) / 2
        self.stat.append(long(locMean))
        x = statistics.mean(self.stat)
        get_stat = self.statisticallyTarget()

        if len(self.stat) == 100:
            self.stat.pop(0)

        if len(self.avg_contours) == 100:
            self.avg_contours.pop(0)
        measure = abs(x - maxLoc[0])

        if measure <= get_stat:
            position = None

            success, box = self.tracker.update(self.frame)
            if self.target_flag and success:
                self.bbox = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                position = (box[0] + int(box[2] / 2), box[1] + int(box[3] / 2))
            else:

                # offers new detection if no selected any for tracking
                if maxLoc[0] != 0 and maxLoc[1] != 0:
                    cv.rectangle(self.frame, (maxLoc[0] - 20, maxLoc[1] - 20), (maxLoc[0] + 20, maxLoc[1] + 20),
                                 (0, 0, 255),
                                 thickness=3)

                self.tempMaxLoc = maxLoc

            if state == 67 or state == 99 or not success:
                print("The target is canceled!")
                self.tracker = cv.legacy_TrackerCSRT.create()
                self.target_flag = False

            # 32 = 'Space' on the keyboard
            if state == 32:
                box = [maxLoc[0] - 20, maxLoc[1] - 20, 40, 40]
                self.tracker.init(self.frame, box)
                self.target_flag = True
                print("On a new target!")

            if position:
                return position

        return -1, -1

    def statisticallyTarget(self):
        average = statistics.mean(self.avg_contours)
        if average >= 1000:
            return 100
        elif average >= 700:
            return 300
        elif average >= 500:
            return 500
        elif average >= 10:
            return 700
        else:
            return 700

    def getFrame(self):
        return self.frame

    def getBox(self):
        if self.bbox:
            return self.bbox
        else:
            return -1, -1, -1, -1
