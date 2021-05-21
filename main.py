import cv2 as cv
import numpy as np
from pygame import mixer


def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def getTarget(frame, cont, sound_flag=None):
    (hc, wc) = frame_resized.shape[:2]  # w:image-width and h:image-height
    cv.circle(frame_resized, (wc // 2, hc // 2), 7, (0, 0, 255), -1)

    M = cv.moments(cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    fX = wc // 2
    fY = hc // 2

    centerX = fX - cX
    centerY = fY - cY

    print(f'({centerX}, {centerY})')
    sound_path = ""

    if centerX > 0 and centerY > 20:  # Up-Left
        cv.arrowedLine(frame_resized, (fX, fY), (fX - 30, fY - 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame_resized, "Up-Left", (30, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        sound_path = "sound/up_left.mp3"

    elif centerX > -10 and centerY >= -20:  # Left
        cv.arrowedLine(frame_resized, (fX, fY), (fX - 30, fY), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame_resized, "Left", (30, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        sound_path = "sound/left.mp3"

    elif centerX > 10 and centerY < -10:  # Down-Left
        cv.arrowedLine(frame_resized, (fX, fY), (fX - 30, fY + 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame_resized, "Down-Left", (30, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        sound_path = "sound/down_left.mp3"

    elif centerX < 0 and centerY < -20:  # Down-Right
        cv.arrowedLine(frame_resized, (fX, fY), (fX + 30, fY + 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame_resized, "Down-Right", (30, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        sound_path = "sound/down_right.mp3"

    elif -50 > centerX and centerY > 50:  # Up-Right
        cv.arrowedLine(frame_resized, (fX, fY), (fX + 30, fY - 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame_resized, "Up-Right", (30, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        sound_path = "sound/up_right.mp3"

    elif centerX < 20 and centerY < -30:  # Down
        cv.arrowedLine(frame_resized, (fX, fY), (fX, fY + 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame_resized, "Down", (30, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        print("down")
        sound_path = "sound/down.mp3"

    else:  # Up
        cv.arrowedLine(frame_resized, (fX, fY), (fX, fY - 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame_resized, "Up", (30, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        sound_path = "sound/up.mp3"

    if sound_flag is not sound_path:
        mixer.music.load(sound_path)
        mixer.music.play()
        sound_flag = sound_path


capture = cv.VideoCapture("videos/1.mp4")
mixer.init()
sound_flag = "none"

# We need to extract the frames one after another
# So on each loop we'll get one frame
while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv.cvtColor(frame_resized, cv.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)

    # a mask allows us to focus only on the parts of the frame that interests us.
    red_mask = cv.inRange(hsvFrame, red_lower, red_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv.dilate(red_mask, kernal)
    res_red = cv.bitwise_and(frame_resized, frame_resized, mask=red_mask)

    # Creating contour to track red color
    # Using contour detection, we can detect the borders of objects, and therefore, localize them easily.
    # RETR_TREE, CHAIN_APPROX_SIMPLE are not really matter, it's just a technic of how to extract the info.
    contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        # Calculate the area and remove the small elements
        area = cv.contourArea(cont)
        pixels = 300
        if len(contours) > 1:
            pixels = 1000

        if area > pixels:
            getTarget(frame_resized, cont, sound_flag)
            x, y, w, h = cv.boundingRect(cont)
            cv.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
            cv.putText(frame_resized, "Red Balloon", (x, y - 10), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))

    cv.imshow("Frame", frame_resized)
    key = cv.waitKey(30)

    # 27 = 'Esc' on the keyboard
    if key == 27:
        break

capture.release()
cv.destroyAllWindows()
