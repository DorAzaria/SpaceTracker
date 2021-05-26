import cv2
import night_detection
import day_detection
import imageio
import numpy as np

lower_night = np.array([0, 0, 0])
upper_night = np.array([179, 255, 155])


def nightModeCheck(hsv) -> bool:
    mask = cv2.inRange(hsv, lower_night, upper_night)
    mean = mask.mean()
    if mean > 250:
        return True
    return False

path = 'videos/night1.mp4'
cap = cv2.VideoCapture(path)
_, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert it to grayscale (easier to find contours)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

if __name__ == "__main__":

    if nightModeCheck(frame):
        print("night mode.")
        night_detection.operate_night_video(path)
    else:
        day_detection.dayAction(path)
        print("day mode.")

    # blur = cv2.blur(gray, (5, 5))  # With kernel size depending upon image size
    # # for pixel in blur:
    # avg = 0
    # for i in range(gray.rows):
    #     for j in range(gray.cols):
    #         avg += gray[i, j]
    # avg /= (gray.rows * gray.cols)
    #
    # if avg < 50:  # 50 is about 20% light, which means night
    #     print('dark picture')
    # #
    # #     # night_detection.operate_night_video('airplain_video.mp4' )
    # else:
    #     # day_detection.operate_day_video("1.mp4")
    #     print('light picture')
