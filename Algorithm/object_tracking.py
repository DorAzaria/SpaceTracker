import cv2
from Algorithm.day_detection import DayMode
from Algorithm.night_detection import NightMode
import numpy as np

"""
    This class manages the whole detecting and tracking algorithm.
    It initially decides whether it is a night / space mode or a day mode.
    After the decision, it sends each frame to the appropriate mode (2 different classes).
    Those classes are detecting objects and when the user decides to track any of the suggest,
    it returns the (X,Y) position of the object, this way, the telescope can also track it.
    
    This class is also responsible to display the graphics and information on the screen.
    Using history of positions, it can suggest by arrows which direction the object moved if it lost.
    And also zooming in to the tracked object if it fully focused by the telescope, this way it can help
    show small objects in space.
"""


class ObjectTracking:
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

    def __init__(self, color_detection=None, color=None):
        self.first_frame = True
        self.mode_flag = None
        self.mode = None
        self.position = None
        self.frame = None
        self.last_time_center = None
        self.last_target = None
        self.color_detection = color_detection
        self.color = color

    """
        This method is actually the main method of the whole algorithm.
        It receives the frame and state from the user.
        It sends each frame to the appropriate mode, and the response is
        the (X,Y) position.

        :param fr - the given frame from the user.
        :param state - the state key (by keyboard) to control the algorithm.
        :return - the (X,Y) position of the tracked object. 
                  If it's not tracking, it will return (-1, -1).
    """

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

        self.GUI(state)

        return self.position

    """
        This method decides if it is a night mode or a day mode.
        Using color range, it get the first frame of the program and then
        it decide if it used for night mode or a day.
        
        :return True if it is a night mode, else it is a day mode. 
    """

    def nightModeCheck(self) -> bool:
        lower_night = np.array([0, 0, 0])
        upper_night = np.array([179, 255, 155])
        mask = cv2.inRange(self.frame, lower_night, upper_night)
        mean = mask.mean()
        if mean > 250:
            self.mode = NightMode()
            return True
        self.mode = DayMode(self.color_detection, self.color)
        return False

    """
        :return the frame to the main program.
    """

    def getFrame(self):
        return self.frame

    """
        This GUI method is responsible to display graphics and data on the screen.
        Using mathematics calculations and frame history it can suggest which direction the object moved 
        if it lost.
        It also display the rectangle of the detected or tracked object, display text data.
        And also, it can zoom-in to the tracked object if the telescope is fully focused on the object.
        
        The method displays an arrow pointing to the direction of tracking objects, if the telescope is right into
        the object, it doesn't show it.
    """

    def GUI(self, state):
        centerX = self.position[0]
        centerY = self.position[1]
        (hc, wc) = self.frame.shape[:2]  # w:image-width and h:image-height
        fX = wc // 2
        fY = hc // 2
        if centerX >= 15 and centerY >= 15:

            box = self.mode.getBox()
            self.drawBox(box)
            limit = self.checkArrowBound((box[0], box[1]), (box[2], box[3]), (fX, fY))
            cv2.putText(self.frame, "TRACKING!", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            self.last_target = centerX, centerY

            if not limit:
                cv2.circle(self.frame, (fX, fY), 2, (0, 0, 255), 4)

                X = int(centerX + (fX - centerX) * 0.5)
                Y = int(centerY + (fY - centerY) * 0.5)
                center = (X, Y)
                self.last_time_center = center
                cv2.arrowedLine(self.frame, (fX, fY), center, (0, 0, 255), thickness=2, tipLength=0.2)
                self.suggestDirection(fX, fY, centerX, centerY)

                # if 'z' or 'Z' then zoom in to the object
                if state == 90 or state == 122:
                    self.zoomInObject(box, wc)

            else:
                cv2.putText(self.frame, "ON TARGET!", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                self.zoomInObject(box, wc)

        else:
            if self.last_time_center:
                cv2.arrowedLine(self.frame, (fX, fY), self.last_time_center, (0, 0, 255), thickness=2, tipLength=0.2)
                if self.last_target:
                    self.suggestDirection(fX, fY, self.last_target[0], self.last_target[1])

            cv2.putText(self.frame, "LOST!", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    """
        This static method decides if the object is in the center of the screen, 
        :return True if the telescope is directed exactly at the object.
    """

    @staticmethod
    def checkArrowBound(bl, tr, p):
        if bl[0] < p[0] < tr[0] and bl[1] < p[1] < tr[1]:
            return True
        else:
            return False

    """
        This method get the border points of the detected object and display it 
        to the screen as a rectangle.
    """

    def drawBox(self, box):
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(self.frame, (x, y), (w, h), (0, 255, 0), 3, 3)

    """
        This method receives the src point (fX,fY) 
        and the dst point (centerX, centerY) of the arrow.
        
        After the mathematics calculation in the GUI method, this method is responsible 
        to the suggestion by very basic 'if' statements.
        It suggest and display the direction of the tracked objects.
    """

    def suggestDirection(self, fX, fY, centerX, centerY):
        if centerX < fX - 10 and centerY < fY - 10:
            cv2.putText(self.frame, "Up-Left", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if centerX < fX and fY - 10 <= centerY < fY + 10:
            cv2.putText(self.frame, "Left", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if centerX < fX - 10 and fY + 10 <= centerY:
            cv2.putText(self.frame, "Down-Left", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if fX - 10 <= centerX < fX + 10 and fY + 20 <= centerY:
            cv2.putText(self.frame, "Down", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if fX - 10 <= centerX < fX + 10 and centerY < fY:
            cv2.putText(self.frame, "Up", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if fX + 10 < centerX and fY + 10 < centerY:
            cv2.putText(self.frame, "Down-Right", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if fX < centerX and fY - 10 < centerY <= fY + 10:
            cv2.putText(self.frame, "Right", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if fX + 10 <= centerX and centerY <= fY - 10:
            cv2.putText(self.frame, "Up-Right", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    """
        This method is useful when the algorithm is tracking after a long distance object
        in the sky.
        It shows at the right-top corner of the screen a zoom-in of the object, only if 
        the telescope is directed exactly at the object.
        This way we can see the object much better and can decide if it is the right 
        object we are interested to track.
        Using linear algebra and geometric calculations. 
    """

    def zoomInObject(self, box, wc):
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        crop_frame = self.frame[y:h, x:w]
        scale_percent = 300  # percent of zoom
        width = int(crop_frame.shape[1] * scale_percent / 100)
        height = int(crop_frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_frame = cv2.resize(crop_frame, dim, interpolation=cv2.INTER_AREA)

        # create an overlay image. You can use any image
        foreground = np.ones((height, width, 3), dtype='uint8')
        alpha = 1
        # Select the region in the background where we want to add the image and add the images using
        # cv2.addWeighted()
        added_image = cv2.addWeighted(resized_frame[0:height, 0:width, :], alpha, foreground[0:height, 0:width, :],
                                      1 - alpha, 0)
        # Change the region with the result
        self.frame[0:height, wc - width:wc + width] = added_image
