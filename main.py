import cv2
from object_tracking import ObjectTracking


def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    path = 'videos/ISS.mp4'
    cap = cv2.VideoCapture(path)
    _, frame = cap.read()

    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_ANY, 30000)

    tracker = ObjectTracking(color_detection=False)

    while capture.isOpened():
        key = cv2.waitKey(30)

        isTrue, frame = capture.read()

        frame = rescaleFrame(frame)
        position = tracker.track(frame, state=key)
        print(position)

        frame = tracker.getFrame()

        cv2.imshow("Space Tracker", frame)

        # 27 = 'Esc' on the keyboard
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
