import cv2
from Algorithm.object_tracking import ObjectTracking


def rescaleFrame(fr, scale=1.0):
    w = int(fr.shape[1] * scale)
    h = int(fr.shape[0] * scale)
    dimensions = (w, h)
    return cv2.resize(fr, dimensions, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    # Define the codec and create VideoWriter object
    path = 'Videos/ISS.mp4'
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_ANY, 0)
    isTrue, frame = capture.read()
    frame = rescaleFrame(frame)
    height, width, channels = frame.shape

    out = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (int(width), int(height)))

    object_tracking = ObjectTracking()

    while capture.isOpened():
        key = cv2.waitKey(30)
        isTrue, frame = capture.read()
        frame = rescaleFrame(frame, scale=0.5)

        position = object_tracking.track(frame, state=key)
        frame = object_tracking.getFrame()
        out.write(frame)

        cv2.imshow("Space Tracker", frame)

        # 27 = 'Esc' on the keyboard
        if key == 27 or key & 0xFF == ord('q'):
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()
