import cv2
import mediapipe
import numpy
import autopy  # for on click event
# this is for right click, scroll, long click
from pynput.mouse import Button, Controller
import os

smoothening = 10

mouse = Controller()

# we take 0 as there is only one camera in our system
cap = cv2.VideoCapture(0)

initHand = mediapipe.solutions.hands  # Initializing mediapipe

# Object of mediapipe with "arguments for the hands module"
mainHand = initHand.Hands(min_detection_confidence=0.8,
                          min_tracking_confidence=0.8)

# Object to draw the connections between each finger index
draw = mediapipe.solutions.drawing_utils

# Outputs the height and width of the screen (1920 x 1080)
wScr, hScr = autopy.screen.size()

pX, pY = 0, 0  # Previous x and y locationq
cX, cY = 0, 0  # Current x and y location


def handLandmarks(colorImg):
    landmarkList = []  # Default values if no landmarks are tracked

    # Object for processing the video input
    landmarkPositions = mainHand.process(colorImg)

    # Stores the out of the processing object (returns False on empty)
    landmarkCheck = landmarkPositions.multi_hand_landmarks
    
    if landmarkCheck:  # Checks if landmarks are tracked
        for hand in landmarkCheck:  # Landmarks for each hand
            # Loops through the 21 indexes and outputs their landmark coordinates (x, y, & z)
            for index, landmark in enumerate(hand.landmark):
                # Draws each individual index on the hand with connections
                draw.draw_landmarks(img, hand, initHand.HAND_CONNECTIONS)
                h, w, c = img.shape  # Height, width and channel on the image
                # Converts the decimal coordinates relative to the image for each index
                centerX, centerY = int(landmark.x * w), int(landmark.y * h)
                # Adding index and its coordinates to a list
                landmarkList.append([index, centerX, centerY])

    return landmarkList


def fingers(landmarks):
    fingerTips = []  # To store 4 sets of 1s or 0s
    tipIds = [4, 8, 12, 16, 20]  # Indexes for the tips of each finger

    # Check if thumb is up
    if landmarks[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
        fingerTips.append(1)
    else:
        fingerTips.append(0)

    # Check if fingers are up except the thumb
    for id in range(1, 5):
        # Checks to see if the tip of the finger is higher than the joint
        if landmarks[tipIds[id]][2] < landmarks[tipIds[id] - 3][2]:
            fingerTips.append(1)
        else:
            fingerTips.append(0)

    return fingerTips


while True:
    check, img = cap.read()  # Reads frames from the camera
    # Changes the format of the frames from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lmList = handLandmarks(imgRGB)
    # cv2.rectangle(img, (75, 75), (640 - 75, 480 - 75), (255, 0, 255), 2)

    if len(lmList) != 0:
        # Gets index 8s x and y values (skips index value because it starts from 1)
        x1, y1 = lmList[8][1:]
        # Gets index 12s x and y values (skips index value because it starts from 1)
        x2, y2 = lmList[12][1:]
        # Calling the fingers function to check which fingers are up
        finger = fingers(lmList)

        # Checks to see if the pointing finger is up and thumb finger is down
        if finger[1] == 1 and finger[2] == 0:
            # Converts the width of the window relative to the screen width
            x3 = numpy.interp(x1, (75, 640 - 75), (0, wScr))
            # Converts the height of the window relative to the screen height
            y3 = numpy.interp(y1, (75, 480 - 75), (0, hScr))

            # Stores previous x locations to update current x location
            cX = pX + (x3 - pX) / 7
            # Stores previous y locations to update current y location
            cY = pY + (y3 - pY) / 7

            # Function to move the mouse to the x3 and y3 values (wSrc inverts the direction)
            autopy.mouse.move(wScr-cX, cY)
            # Stores the current x and y location as previous x and y location for next loop
            pX, pY = cX, cY

        # Checks to see if the pointer finger is down and thumb finger is up
        if finger[1] == 1 and finger[2] == 1:
            autopy.mouse.click()  # Left click

        if finger[1] == 0 and finger[4] == 1:
            mouse.click(Button.right, 1)

        if finger[2] == 1:
            mouse.scroll(0, -2)

        if finger[2] == 0:
            mouse.scroll(0, 2)

        # if fingers[0] == 1:
        #     os.system("shutdown /r /t 1")

        # if fingers[0] == 0 or fingers[1] == 0 or fingers[2] == 0 or fingers[3] == 0 or fingers[4] == 0:
        #     os.system("shutdown /s /t 1")

        if finger[1] == 1 and finger[3] == 1:
            cv2.waitKey(1)
            break

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
