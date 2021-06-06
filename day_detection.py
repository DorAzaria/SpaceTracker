import cv2 as cv
import numpy as np
from numpy import long
import statistics

# Set range for blue color for moving to sky mode
# define mask
lower_blue = np.array([0, 0, 0])
upper_blue = np.array([179, 255, 155])
stat = []
avg_contours = []
tempMaxLoc = (0, 0)
backSub = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=True)
backSub.setNMixtures(8)
lastMean = 0
tracker = cv.legacy_TrackerCSRT.create()
target_flag = False


def statisticallyTarget():
    average = statistics.mean(avg_contours)
    if average >= 1000:
        return 100
    elif average >= 700:
        return 300
    elif average >= 500:
        return 500
    elif average >= 10:
        return 700
    else:
        return 700


def skyModeCheck(hsv) -> bool:
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    mean = mask.mean()
    if mean < 20.0:
        return True
    return False


def rescaleFrame(scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def getTarget(fr, cont):
    (hc, wc) = fr.shape[:2]  # w:image-width and h:image-height

    M = cv.moments(cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    fX = wc // 2
    fY = hc // 2
    centerX = fX - cX
    centerY = fY - cY

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


def groundMode(frame_resized, hsvFrame):
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

    # Creating contour to track red color
    # Using contour detection, we can detect the borders of objects, and therefore, localize them easily.
    # RETR_TREE, CHAIN_APPROX_SIMPLE are not really matter, it's just a technic of how to extract the info.
    contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: cv.contourArea(x))
    if contours:
        aa = contours[-1]
        getTarget(frame_resized, aa)
        x, y, w, h = cv.boundingRect(aa)
        cv.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle(img, (x, y), ((x + w), (y + h)), (0, 255, 0), 3, 3)
    cv.putText(img, "ON TARGET!", (5, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))


def skyMode(fr):
    global tempMaxLoc
    global lastMean
    global target_flag
    global tracker

    fgMask = backSub.apply(fr)

    fram, thresh = cv.threshold(fgMask, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    avg = len(contours)
    avg_contours.append(avg)

    gray = cv.GaussianBlur(fgMask, (7, 7), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    locMean = (maxLoc[0] + maxLoc[1]) / 2
    stat.append(long(locMean))
    x = statistics.mean(stat)
    get_stat = statisticallyTarget()

    if len(stat) == 100:
        stat.pop(0)
        lastMean = x

    if len(avg_contours) == 100:
        avg_contours.pop(0)
    measure = abs(x - maxLoc[0])
    key = cv.waitKey(1)

    if measure <= get_stat:
        success, bbox = tracker.update(fr)
        if target_flag and success:
            drawBox(fr, bbox)
        else:

            # 32 = 'Space' on the keyboard
            if key == 32:
                box = [maxLoc[0] - 20, maxLoc[1] - 20, 40, 40]
                tracker.init(fr, box)
                target_flag = True
                print("On a new target!")

            if maxLoc[0] != 0 and maxLoc[1] != 0:
                cv.rectangle(fr, (maxLoc[0] - 20, maxLoc[1] - 20), (maxLoc[0] + 20, maxLoc[1] + 20), (0, 0, 255),
                             thickness=3)

            cv.putText(fr, "LOST!", (5, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            tempMaxLoc = maxLoc

        if key == 67 or key == 99:
            print("The target is canceled!")
            tracker = cv.legacy_TrackerCSRT.create()
            target_flag = False


def dayAction(path):
    global frame

    capture = cv.VideoCapture(path)
    capture.set(cv.CAP_ANY, 90000)

    while capture.isOpened():
        timer = cv.getTickCount()
        isTrue, frame = capture.read()
        frame_resized = rescaleFrame()

        hsvFrame = cv.cvtColor(frame_resized, cv.COLOR_BGR2HSV)

        if skyModeCheck(hsvFrame):
            cv.putText(frame_resized, "Sky Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            skyMode(frame_resized)
        else:
            cv.putText(frame_resized, "Ground Mode", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            groundMode(frame_resized, hsvFrame)

        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        cv.putText(frame_resized, f"FPS={int(fps)}", (5, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("Frame", frame_resized)

        key = cv.waitKey(30)

        # 27 = 'Esc' on the keyboard
        if key == 27:
            break

    capture.release()
    cv.destroyAllWindows()
