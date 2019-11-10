import cv2 as cv
import numpy as np

#CAMERA_SETUP
CAMERA_SOURCE = 0
def cameraSetup():
    cap = cv.VideoCapture(CAMERA_SOURCE)
    if cap is None or not cap.isOpened():
        print("Unable to open camera. Source: ", CAMERA_SOURCE)
        quit()
    return cap
#/CAMERA_SETUP

#TRACKBAR_SETUP
def trackbarCall(x):
    pass
def trackbarSetup():
    cv.namedWindow("Trackbars")
    cv.createTrackbar("LH", "Trackbars", 80, 255, trackbarCall)
    cv.createTrackbar("UH", "Trackbars", 135, 255, trackbarCall)
    cv.createTrackbar("LL", "Trackbars", 80, 255, trackbarCall)
    cv.createTrackbar("UL", "Trackbars", 155, 255, trackbarCall)
    cv.createTrackbar("LS", "Trackbars", 20, 255, trackbarCall)
    cv.createTrackbar("US", "Trackbars", 180, 255, trackbarCall)
    
    cv.createTrackbar("widthScale", "Trackbars", 200, 1000, trackbarCall)
    cv.createTrackbar("heightScale", "Trackbars", 400, 1000, trackbarCall)

#/TRACKBAR_SETUP
    
cap = cameraSetup()
trackbarSetup()
while True:
    _, frame = cap.read()
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    lowerHLSThreshold =np.array([cv.getTrackbarPos("LH", "Trackbars"), cv.getTrackbarPos("LL", "Trackbars"), cv.getTrackbarPos("LS", "Trackbars")])
    upperHLSThreshold = np.array([cv.getTrackbarPos("UH", "Trackbars"), cv.getTrackbarPos("UL", "Trackbars"), cv.getTrackbarPos("US", "Trackbars")])
    hlsFiltered = cv.inRange(hls, lowerHLSThreshold, upperHLSThreshold)
    
    imgWidth = frame.shape[1]
    imgHeight = frame.shape[0]
    
    #Construct the indexes of a trapezoid as the homography:
    widthScale = cv.getTrackbarPos("widthScale", "Trackbars") / 1000.0
    heightScale = cv.getTrackbarPos("heightScale", "Trackbars") / 1000.0
    offset = imgWidth / 2
    initialPoints = np.float32([[int(widthScale * imgWidth), int(heightScale * imgHeight)],
                                [int((1 - widthScale) * imgWidth), int(heightScale * imgHeight)],
                                [0, imgHeight],
                                [imgWidth, imgHeight]])
    finalPoints = np.float32([[offset,  0],
                              [offset + imgWidth, 0],
                              [offset, imgHeight],
                              [offset + imgWidth, imgHeight]])
    M = cv.getPerspectiveTransform(initialPoints, finalPoints)
    warp = cv.warpPerspective(frame, M, (2*imgWidth, imgHeight))
    
    cv.imshow("frame", frame)
    cv.imshow("hlsFiltered", hlsFiltered)
    cv.imshow("warp", warp)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        cap.release()
        print("Program exiting.")
        quit()
