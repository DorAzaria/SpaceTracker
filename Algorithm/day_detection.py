import cv2 as cv
import numpy as np
from numpy import long
import statistics

"""
    This class manages the detecting and tracking algorithm in day-light.
    Using smart algorithms of OpenCV and some original methods.
    In each frame, it decides if it is a ground or sky mode, it way
    it can priority the detection by the number of objects in the frame and
    then can work much better and right.
"""


class DayMode:
    """
        The constructor can be non-parametric, in this case the detection
        of objects in ground mode will be calculated by speed detection.

        :param color_detection - if the constructor receives a True 'color_detection',
                  the detection of objects in ground mode will be calculated by color detection.
                  (the default is RED).

        :param color - if given 'color', it can detect objects by color in ground mode.
                  The options are 'RED', 'ORANGE' and 'YELLOW'.
                  If it's empty, it will detect 'RED' colors.
    """

    def __init__(self, color_detection, color=None):
        self.stat = []
        self.avg_contours = []
        self.tempMaxLoc = (0, 0)
        self.backSub = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)
        self.backSub.setNMixtures(8)
        self.tracker = cv.legacy_TrackerCSRT.create()
        self.target_flag = False
        self.frame = None
        self.bbox = None
        self.last_mode = None
        self.position = None
        self.color_detection = color_detection
        self.color = color

        # Set range for red color
        self.red_lower = np.array([136, 87, 111], np.uint8)
        self.red_upper = np.array([180, 255, 255], np.uint8)

        # Set range for yellow color
        self.yellow_lower = np.array([16, 175, 237], np.uint8)
        self.yellow_upper = np.array([65, 255, 255], np.uint8)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        self.kernal = np.ones((5, 5), "uint8")

    """
        This method using the skyModeCheck() for using the appropriate method.
    
        :param fr - the given frame from the user.
        :param state - the state key (by keyboard) to control the algorithm.
        :return - the (X,Y) position of the tracked object. 
                  If it's not tracking, it will return (-1, -1).
    """

    def dayAction(self, fr, state) -> tuple:
        self.frame = fr

        if self.skyModeCheck():
            cv.putText(self.frame, "Sky Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            self.position = self.skyMode(state)
        elif self.color_detection:
            cv.putText(self.frame, "Ground Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            self.position = self.groundModeByColor(state)
        else:  # ground without color mode:
            cv.putText(self.frame, "Ground Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            self.position = self.groundMode(state)

        return self.position

    """
        This method decides using color range if the frame is at ground or sky mode.
        
        :return True if it's sky mode.
    """

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
                
        The transition from sky to ground (and vice-versa) is safe while we track an object even though the two 
        algorithms are completely different, it is a safe-change between algorithms without losing the object.
                
        :param state - the state key (by keyboard) to control the algorithm.
        
        :return a tuple of (X,Y) position of the detected/tracked object.
    """
    def skyMode(self, state) -> tuple:

        if self.last_mode == 'ground' and self.position[0] != -1 and self.position[1] != -1:
            box = [self.bbox[0], self.bbox[1], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]]
            self.tracker.init(self.frame, box)
            self.target_flag = True
            self.last_mode = 'sky'
            return self.bbox[0] + int(self.bbox[2] / 2), self.bbox[1] + int(self.bbox[3] / 2)

        self.last_mode = 'sky'

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
                self.bbox = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                position = (box[0] + int(box[2] / 2), box[1] + int(box[3] / 2))
            else:
                # offers new detection if no selected any for tracking
                if maxLoc[0] != 0 and maxLoc[1] != 0:
                    cv.rectangle(self.frame, (maxLoc[0] - 20, maxLoc[1] - 20), (maxLoc[0] + 20, maxLoc[1] + 20),
                                 (0, 0, 255), thickness=3)

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

    """
        This method manages the detection and tracking of an object in the ground
        using color detection.
        
        About the algorithm:
            Given a color by the user (RED, YELLOW or ORANGE).
            The algorithm detects the object by color.
            Using color range of the given color, it detect the most 
            stronger color and relevant-sized object.
            
            Using HSV color picture we search in each frame the correct color we wish to detect
            and then using mask we find the right contours (shapes).
            
           When the program detect an object, the user can press specific keys to control the state.
           Using smart tracker algorithm of CSRT -(Discriminative Correlation Filter with Channel and 
           Spatial Reliability) which is specializing in fast-moving objects.
           
           The user can press:
                - 'space' to track a detected object.
                - 'c' to cancel the tracking and move to detection scenario.
                
        The transition from sky to ground (and vice-versa) is safe while we track an object even though the two 
        algorithms are completely different, it is a safe-change between algorithms without losing the object.
                
        :param state - the state key (by keyboard) to control the algorithm.
        
        :return a tuple of (X,Y) position of the detected/tracked object.
            
    """

    def groundModeByColor(self, state) -> tuple:

        if self.last_mode == 'sky' and self.position[0] != -1 and self.position[1] != -1:
            box = [self.bbox[0], self.bbox[1], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]]
            self.tracker.init(self.frame, box)
            self.target_flag = True
            self.last_mode = 'ground'
            return self.bbox[0] + int(self.bbox[2] / 2), self.bbox[1] + int(self.bbox[3] / 2)

        self.last_mode = 'ground'
        hsvFrame = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)

        mask = None
        # a mask allows us to focus only on the parts of the frame that interests us.
        if self.color is None or self.color == 'RED' or self.color == 'ORANGE':
            mask = cv.inRange(hsvFrame, self.red_lower, self.red_upper)

        if self.color == 'YELLOW':
            mask = cv.inRange(hsvFrame, self.yellow_lower, self.yellow_upper)

        mask = cv.dilate(mask, self.kernal)

        # Creating contour to track red color
        # Using contour detection, we can detect the borders of objects, and therefore, localize them easily.
        # RETR_TREE, CHAIN_APPROX_SIMPLE are not really matter, it's just a technic of how to extract the info.
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours.sort(key=lambda t: cv.contourArea(t))
        if contours:
            x, y, w, h = cv.boundingRect(contours[-1])
            success, box = self.tracker.update(self.frame)

            if self.target_flag and success:
                self.bbox = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                position = (box[0] + int(box[2] / 2), box[1] + int(box[3] / 2))
            else:
                self.bbox = (x, y, x + w, y + h)
                if x != 0 and y != 0:
                    cv.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
                position = -1, -1

            if state == 67 or state == 99 or not success:
                self.tracker = cv.legacy_TrackerCSRT.create()
                self.target_flag = False

            # 32 = 'Space' on the keyboard
            if state == 32:
                box = [x, y, w, h]
                self.tracker.init(self.frame, box)
                self.target_flag = True

            return position

        return -1, -1

    """
        This method manages the detection and tracking of an object in the ground
        by moving object search.
        Used in case we don't want to use the color-detection mode.
        
        About the algorithm:
            Using MOG2 algorithm we can detect the moving objects.
            Then we picked the fastest one (which is presented by the strongest white color).
            The detect object is displayed to the screen and the user can chose to track it or not.
            
           When the program detect an object, the user can press specific keys to control the state.
           Using smart tracker algorithm of CSRT -(Discriminative Correlation Filter with Channel and 
           Spatial Reliability) which is specializing in fast-moving objects.
       
           The user can press:
                - 'space' to track a detected object.
                - 'c' to cancel the tracking and move to detection scenario.
                
        The transition from sky to ground (and vice-versa) is safe while we track an object even though the two 
        algorithms are completely different, it is a safe-change between algorithms without losing the object.
                
        :param state - the state key (by keyboard) to control the algorithm.
        
        :return a tuple of (X,Y) position of the detected/tracked object.
    
    """
    def groundMode(self, state) -> tuple:

        if self.last_mode == 'sky' and self.position[0] != -1 and self.position[1] != -1:
            box = [self.bbox[0], self.bbox[1], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]]
            self.tracker.init(self.frame, box)
            self.target_flag = True
            self.last_mode = 'ground'
            return self.bbox[0] + int(self.bbox[2] / 2), self.bbox[1] + int(self.bbox[3] / 2)

        self.last_mode = 'ground'

        fgMask = self.backSub.apply(self.frame)

        gray = cv.GaussianBlur(fgMask, (7, 7), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
        success, box = self.tracker.update(self.frame)
        position = None
        if self.target_flag and success:
            self.bbox = (box[0], box[1], box[0] + box[2], box[1] + box[3])
            position = (box[0] + int(box[2] / 2), box[1] + int(box[3] / 2))
        else:
            # offers new detection if no selected any for tracking
            if maxLoc[0] != 0 and maxLoc[1] != 0:
                cv.rectangle(self.frame, (maxLoc[0] - 20, maxLoc[1] - 20), (maxLoc[0] + 20, maxLoc[1] + 20),
                             (0, 0, 255), thickness=3)

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

    def getFrame(self):
        return self.frame

    def getBox(self):
        if self.bbox:
            return self.bbox
        else:
            return -1, -1, -1, -1
