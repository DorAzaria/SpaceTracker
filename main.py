import cv2
import night_detection
import day_detection
import numpy as np

lower_night = np.array([0, 0, 0])
upper_night = np.array([179, 255, 155])


def nightModeCheck(hsv) -> bool:
    mask = cv2.inRange(hsv, lower_night, upper_night)
    mean = mask.mean()
    if mean > 250:
        return True
    return False


path = 'videos/9.MTS'
cap = cv2.VideoCapture(path)
_, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert it to grayscale (easier to find contours)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

if __name__ == "__main__":

    if nightModeCheck(frame):
        print("Night Mode.")
        night_detection.operate_night_video(path)
    else:
        print("Day Mode.")
        day_detection.dayAction(path)
