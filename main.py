import cv2
from serial import SerialException
from Algorithm.object_tracking import ObjectTracking
import time
from Telescope import Telecontrol


class SpaceTracker:

    def __init__(self, telescopeEnabled=True, port=None):
        self.telescopeEnabled = telescopeEnabled
        self.capture = None
        self.frame = None
        self.out = None
        self.object_tracking = None

        if self.telescopeEnabled:
            if port is None:
                print('Must insert a port with the constructor!')
                exit()
            try:
                self.telescope = Telecontrol.Telcontrol(port)
                time.sleep(2)
            except SerialException as error:
                print('Telescope is not connected!')
                print(f'Error details: {error}')
                exit()

    def start(self, video_path):
        self.capture = cv2.VideoCapture(video_path)
        self.capture.set(cv2.CAP_ANY, 0)
        isTrue, self.frame = self.capture.read()
        self.rescaleFrame(scale=0.5)
        height, width, channels = self.frame.shape
        self.out = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                   (int(width), int(height)))
        self.object_tracking = ObjectTracking()

        while self.capture.isOpened():
            key = cv2.waitKey(30)
            isTrue, self.frame = self.capture.read()
            self.rescaleFrame(scale=0.5)

            position = self.object_tracking.track(self.frame, state=key)
            self.frame = self.object_tracking.getFrame()
            self.moveTelescope(position)
            self.out.write(self.frame)

            cv2.imshow("Space Tracker", self.frame)

            # 27 = 'Esc' on the keyboard
            if key == 27 or key & 0xFF == ord('q'):
                break

        self.capture.release()
        self.out.release()
        cv2.destroyAllWindows()

    def moveTelescope(self, position):
        if self.telescopeEnabled and position[0] != -1 and position[1] != -1:
            dx = position[0] - (self.frame.shape[1] // 2)
            dy = position[1] - (self.frame.shape[0] // 2)
            sx = 4
            sy = 4

            if abs(dx) < 100:
                sx = 3
            if abs(dx) < 75:
                sx = 2
            if abs(dx) < 50:
                sx = 1
            if abs(dx) < 10:
                sx = 0

            if abs(dy) < 100:
                sx = 3
            if abs(dy) < 75:
                sx = 2
            if abs(dy) < 50:
                sx = 1
            if abs(dy) < 10:
                sx = 0

            telescope_data = "dx: " + str(round(dx)) + ", dy: " + str(
                round(-dy)) + ", tele speed x axis: " + str(sy) + ", tele speed y axis: " + str(sx)
            cv2.putText(self.frame, telescope_data, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (120, 120, 20), 2)

            self.telescope.moveY(direction=round(dx), speed=sx)
            self.telescope.moveX(direction=round(-dy), speed=sy)

    def rescaleFrame(self, scale=1.0):
        w = int(self.frame.shape[1] * scale)
        h = int(self.frame.shape[0] * scale)
        dimensions = (w, h)
        self.frame = cv2.resize(self.frame, dimensions, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    tracker = SpaceTracker(telescopeEnabled=False)
    path = 0
    tracker.start(video_path=path)
