import cv2 as cv
from numpy import long
import statistics

"""
    This class manages the detecting and tracking algorithm in night/space.
    Using smart algorithms of OpenCV and some original methods.
"""


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

    """
        :return a tuple (X,Y) of the position of the detected/tracked object.
    """
    def nightAction(self, fr, state) -> tuple:
        self.frame = fr
        cv.putText(self.frame, "Sky Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        return self.skyMode(state)

    """
        This method manages the detection and tracking of an object in the sky.

        About the algorithm:
            Using background subtraction algorithm of Gaussian Mixture-based Background Segmentation,
            with a very short history it can detect moving objects.
            It generates a BnW frame, the whiter, the faster.

            We used another algorithm to detect the whiter object and then analyse it for correctness.
            After detecting, we used some limited-data-structures that contains:
                1) 'stat' - the position of an object in each frame.
                            If the new position of the detected object is around the mean probability of 'stat' list
                            then it probably an object that moves logically.
                2) 'avg_contours' - the number of total objects we can get in each frame.
                                    By probabilistic calculation of variance and mean this list can decide 
                                    if the picture is 'loud' by many objects (it happen mostly near the ground).
                                    So if the variance and mean is big, the position of the detected object
                                    can't go far from the last position it was because there's too many objects.
                                    If the variance and mean is small, we can look for a far position of the object 
                                    comparing to its last position.

           When the program detect an object, the user can press specific keys to control the state.
           Using smart tracker algorithm of CSRT -(Discriminative Correlation Filter with Channel and 
           Spatial Reliability) which is specializing in fast-moving objects.

           The user can press:
                - 'space' to track a detected object.
                - 'c' to cancel the tracking and move to detection scenario.

        :param state - the state key (by keyboard) to control the algorithm.

        :return a tuple of (X,Y) position of the detected/tracked object.
    """
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
                self.tracker = cv.legacy_TrackerCSRT.create()
                self.target_flag = False

            # 32 = 'Space' on the keyboard
            if state == 32:
                box = [maxLoc[0] - 20, maxLoc[1] - 20, 40, 40]
                self.tracker.init(self.frame, box)
                self.target_flag = True

            if position:
                return position

        return -1, -1

    """
        This method suggest the size of radius of detecting-search
        using the mean of the history list of the number of moving objects we found
        in each frame.
        If the number if big, then look somewhere near because we can't predict where the object
        has lost if there is too many objects around.

        If the number is small, then it can search by a big radius for lost the object.

        :return an integer value (the radius).
    """

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
