import cv2 as cv
import numpy as np
from numpy import long
import statistics


class DayMode:

    def __init__(self):
        self.stat = []
        self.avg_contours = []
        self.tempMaxLoc = (0, 0)
        self.backSub = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)
        self.backSub.setNMixtures(8)
        self.tracker = cv.legacy_TrackerCSRT.create()
        self.target_flag = False
        self.frame = None
        self.bbox = None

    def dayAction(self, fr, state) -> tuple:
        self.frame = fr
        timer = cv.getTickCount()

        if self.skyModeCheck():
            cv.putText(self.frame, "Sky Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            position = self.skyMode(state)
        else:
            cv.putText(self.frame, "Ground Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            position = self.groundMode(state)

        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        cv.putText(self.frame, f"FPS={int(fps)}", (5, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        return position

    def skyModeCheck(self) -> bool:
        hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)

        # Set range for blue color for moving to sky mode
        # define mask
        lower_blue = np.array([0, 0, 0])
        upper_blue = np.array([179, 255, 155])

        mask = cv.inRange(hsv, lower_blue, upper_blue)
        mean = mask.mean()
        if mean < 20.0:
            return True
        return False

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
            success, box = self.tracker.update(self.frame)
            position = None
            if self.target_flag and success:
                self.bbox = (box[0], box[1], box[0]+box[2], box[1]+box[3])
                position = (box[0] + int(box[2] / 2), box[1] + int(box[3] / 2))
            else:

                # 32 = 'Space' on the keyboard
                if state == 32:
                    box = [maxLoc[0] - 20, maxLoc[1] - 20, 40, 40]
                    self.tracker.init(self.frame, box)
                    self.target_flag = True
                    print("On a new target!")

                if maxLoc[0] != 0 and maxLoc[1] != 0:
                    cv.rectangle(self.frame, (maxLoc[0] - 20, maxLoc[1] - 20), (maxLoc[0] + 20, maxLoc[1] + 20),
                                 (0, 0, 255), thickness=3)

                cv.putText(self.frame, "LOST!", (5, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                self.tempMaxLoc = maxLoc

            if state == 67 or state == 99:
                print("The target is canceled!")
                self.tracker = cv.legacy_TrackerCSRT.create()
                self.target_flag = False

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

    def groundMode(self, state) -> tuple:
        hsvFrame = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)

        # Set range for red color and
        # define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        # a mask allows us to focus only on the parts of the frame that interests us.
        red_mask = cv.inRange(hsvFrame, red_lower, red_upper)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernal = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv.dilate(red_mask, kernal)

        # Creating contour to track red color
        # Using contour detection, we can detect the borders of objects, and therefore, localize them easily.
        # RETR_TREE, CHAIN_APPROX_SIMPLE are not really matter, it's just a technic of how to extract the info.
        contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda x: cv.contourArea(x))
        if contours:
            aa = contours[-1]
            x, y, w, h = cv.boundingRect(aa)
            cv.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
            self.bbox = (x, y, x + w, y + h)
            return x, y

        return -1, -1

    def getFrame(self):
        return self.frame

    def getBox(self):
        if self.bbox:
            return self.bbox
        else:
            return -1, -1, -1, -1
