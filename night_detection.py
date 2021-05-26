import cv2


def operate_night_video(input_image):

    cap = cv2.VideoCapture(input_image)
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 5.0, (1280, 720))

    # read first two frames
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    print(frame1.shape)
    while cap.isOpened():

        # prepare mask:
        diff = cv2.absdiff(frame1, frame2)  # find the difference between the frames
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # convert it to grayscale (easier to find contours)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # divide to black and white (if above 20, it's white, otherwise black)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print_line = 20
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) > 85:
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
                cv2.putText(frame1, "X: {} Y: {}".format(x, y), (10, print_line), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)
                # print([x, y, w, h])
                print_line = print_line + 40
        image = cv2.resize(frame1, (1280, 720))
        out.write(image)
        cv2.imshow("feed", frame1)
        frame1 = frame2
        _, frame2 = cap.read()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()