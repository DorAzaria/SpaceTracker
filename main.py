import cv2
from Algorithm.object_tracking import ObjectTracking


def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    # Define the codec and create VideoWriter object
    path = 'videos/night2.mp4'
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_ANY, 0)
    isTrue, frame = capture.read()
    frame = rescaleFrame(frame)
    height, width, channels = frame.shape

    out = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (int(width), int(height)))

    tracker = ObjectTracking(color_detection=False)

    while capture.isOpened():
        key = cv2.waitKey(30)
        isTrue, frame = capture.read()
        frame = rescaleFrame(frame)

        position = tracker.track(frame, state=key)
        print(position)
        frame = tracker.getFrame()
        out.write(frame)

        cv2.imshow("Space Tracker", frame)

        # 27 = 'Esc' on the keyboard
        if key == 27 or key & 0xFF == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()
