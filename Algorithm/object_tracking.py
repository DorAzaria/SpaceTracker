import cv2
from Algorithm.day_detection import DayMode
from Algorithm.night_detection import NightMode
import numpy as np


class ObjectTracking:

    def __init__(self, color_detection):
        self.first_frame = True
        self.mode_flag = None
        self.mode = None
        self.position = None
        self.frame = None
        self.last_time_center = None
        self.last_target = None
        self.color_detection = color_detection

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
        self.mode = DayMode(self.color_detection)
        return False

    def getFrame(self):
        return self.frame

    def GUI(self):
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

            else:
                cv2.putText(self.frame, "ON TARGET!", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                crop_frame = self.frame[y:h, x:w]
                scale_percent = 300  # percent of original size
                width = int(crop_frame.shape[1] * scale_percent / 100)
                height = int(crop_frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                resized_frame = cv2.resize(crop_frame, dim, interpolation=cv2.INTER_AREA)

                # create an overlay image. You can use any image
                foreground = np.ones((height, width, 3), dtype='uint8')
                alpha = 1
                # Select the region in the background where we want to add the image and add the images using
                # cv2.addWeighted()0:100
                added_image = cv2.addWeighted(resized_frame[0:height, 0:width, :], alpha, foreground[0:height, 0:width, :],
                                              1 - alpha, 0)
                # Change the region with the result
                self.frame[0:height, wc-width:wc+width] = added_image

        else:
            if self.last_time_center:
                cv2.arrowedLine(self.frame, (fX, fY), self.last_time_center, (0, 0, 255), thickness=2, tipLength=0.2)
                if self.last_target:
                    self.suggestDirection(fX,fY,self.last_target[0], self.last_target[1])

            cv2.putText(self.frame, "LOST!", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    @staticmethod
    def checkArrowBound(bl, tr, p):
        if bl[0] < p[0] < tr[0] and bl[1] < p[1] < tr[1]:
            return True
        else:
            return False

    def drawBox(self, box):
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(self.frame, (x, y), (w, h), (0, 255, 0), 3, 3)

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
