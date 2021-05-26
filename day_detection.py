import cv2 as cv
import numpy as np
from pygame import mixer

# Set range for blue color for moving to sky mode
# define mask
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([179, 255, 155])


def skyModeCheck(hsv) -> bool:
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    mean = mask.mean()
    if mean < 20.0:
        return True
    return False


def rescaleFrame(scale=0.):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def getTarget(fr, cont):
    (hc, wc) = fr.shape[:2]  # w:image-width and h:image-height
    # cv.circle(fr, (wc // 2, hc // 2), 7, (0, 0, 255), -1)

    M = cv.moments(cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    fX = wc // 2
    fY = hc // 2

    centerX = fX - cX
    centerY = fY - cY
    cv.putText(fr, f"Center({abs(centerX)},{abs(centerY)})", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # print(f'({centerX}, {centerY})')
    sound_path = ""

    if centerX > 0 and centerY > 20:  # Up-Left
        # cv.arrowedLine(frame_resized, (fX, fY), (fX - 30, fY - 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame, "Up-Left", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    elif centerX > -10 and centerY >= -20:  # Left
        # cv.arrowedLine(frame_resized, (fX, fY), (fX - 30, fY), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame, "Left", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    elif centerX > 10 and centerY < -10:  # Down-Left
        # cv.arrowedLine(frame_resized, (fX, fY), (fX - 30, fY + 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame, "Down-Left", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    elif centerX < 0 and centerY < -20:  # Down-Right
        # cv.arrowedLine(frame_resized, (fX, fY), (fX + 30, fY + 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame, "Down-Right", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    elif -50 > centerX and centerY > 50:  # Up-Right
        # cv.arrowedLine(frame_resized, (fX, fY), (fX + 30, fY - 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame, "Up-Right", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    elif centerX < 20 and centerY < -30:  # Down
        # cv.arrowedLine(frame_resized, (fX, fY), (fX, fY + 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame, "Down", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    else:  # Up
        # cv.arrowedLine(frame_resized, (fX, fY), (fX, fY - 30), (0, 0, 255), thickness=2, tipLength=0.5)
        cv.putText(frame, "Up", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


    # if sound_flag is not sound_path:
    #     mixer.music.load(sound_path)
    #     mixer.music.play()
    #     sound_flag = sound_path


def groundMode(hsvFrame):
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
    res_red = cv.bitwise_and(frame, frame, mask=red_mask)

    # Creating contour to track red color
    # Using contour detection, we can detect the borders of objects, and therefore, localize them easily.
    # RETR_TREE, CHAIN_APPROX_SIMPLE are not really matter, it's just a technic of how to extract the info.
    contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: cv.contourArea(x))
    if contours:
        aa = contours[-1]
        getTarget(frame, aa)
        x, y, w, h = cv.boundingRect(aa)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)


def skyMode(curr_frame, hsvFrame):

    gray = cv.cvtColor(curr_frame, cv.COLOR_RGB2GRAY)
    _, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)

    # Creating contour to track red color
    # Using contour detection, we can detect the borders of objects, and therefore, localize them easily.
    # RETR_TREE, CHAIN_APPROX_SIMPLE are not really matter, it's just a technic of how to extract the info.
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: cv.contourArea(x), reverse=True)
    if contours:
        k = 0
        for i in range(len(contours)):
            aa = contours[i]

            if k <= 5 and 5 < cv.contourArea(aa) < 100000:
                if len(contours) == 1:
                    getTarget(frame, aa)
                x, y, w, h = cv.boundingRect(aa)
                cv.rectangle(frame, (x - 10, y - 10), (x + w, y + h), (0, 255, 0), thickness=3)
            k += 1
    groundMode(hsvFrame)


def dayAction(path):
    # global frame_resized
    global frame

    capture = cv.VideoCapture(path)
    capture.set(cv.CAP_ANY, 75000)

    mixer.init()
    sound_flag = "none"

    # We need to extract the frames one after another
    # So on each loop we'll get one frame
    while capture.isOpened():
        isTrue, frame = capture.read()
        # frame_resized = rescaleFrame()

        # Convert the imageFrame in
        # BGR(RGB color space) to
        # HSV(hue-saturation-value)
        # color space
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        if skyModeCheck(hsvFrame):
            cv.putText(frame, "Sky Mode", (5, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            skyMode(frame,hsvFrame)
        else:
            cv.putText(frame, "Ground Mode", (5, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            groundMode(hsvFrame)

        cv.imshow("Frame", frame)

        key = cv.waitKey(30)

        # 27 = 'Esc' on the keyboard
        if key == 27:
            break

    capture.release()
    cv.destroyAllWindows()
