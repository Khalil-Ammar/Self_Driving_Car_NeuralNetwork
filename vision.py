import cv2 as cv 
import numpy as np
#import RPi.GPIO as gpio

###CONSTANTS
#IO Pin Declarations
#/IO Pin Declarations

#Run-time options
USER_DEFINES_HLS = False
USER_DEFINES_PERSPECTIVE = False
CAMERA_SOURCE = 0

SHOW_HLS_WINDOW = True
SHOW_COLOUR_FILTER = True
SHOW_WARP_WINDOW = True
SHOW_WARP_BINARY = True
#/Run-time options

#Image Processing Constants
IMG_RESIZE_SCALE = 3
#/Image Processing Constants
###/CONSTANTS

###MAIN FUNCTIONS
#PROCESS_IMAGE
def process_image(img):
    #The purpose of this function is to transform an image from the car's camera feed into a top-down,
    #black-and-white image containing only lane lines
    #
    #The main steps of this function are:
    #1. Color filtering
    #2. Edge detection(?)
    #3. Perspective transform
    #4. Binarization
    hwc = img.shape
    height = hwc[0]
    width = hwc[1]
    img = cv.resize(img, (int(width/IMG_RESIZE_SCALE), int(height/IMG_RESIZE_SCALE)), interpolation = cv.INTER_AREA)
    hwc = img.shape
    height = hwc[0]
    width = hwc[1]
    #Color filtering:
    
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    if SHOW_HLS_WINDOW:
        cv.imshow("HLS", hls)
    if USER_DEFINES_HLS:
        lt = np.array([cv.getTrackbarPos("LH", "Trackbars"), cv.getTrackbarPos("LL", "Trackbars"), cv.getTrackbarPos("LS", "Trackbars")])
        ut = np.array([cv.getTrackbarPos("UH", "Trackbars"), cv.getTrackbarPos("UL", "Trackbars"), cv.getTrackbarPos("US", "Trackbars")])
    else:
        lt = np.array([90,60,40])
        ut = np.array([120,180,160])
    mask = cv.inRange(hls, lt, ut)
    res = cv.bitwise_and(img, img, mask=mask)
    if SHOW_COLOUR_FILTER:
        cv.imshow("COLOUR FILTER", res)
    #/Color filtering
    
    #Perspective Transform:
    if USER_DEFINES_PERSPECTIVE:
        ps = cv.getTrackbarPos("PS", "Trackbars") / 1000
        psh = cv.getTrackbarPos("PSH", "Trackbars") / 1000
    else:
        ps = 0.2
        psh = 0.4
    tl = [ps * width, psh*height]
    tr = [(1-ps)*width, psh*height]
    br = [width, height]
    bl = [0, height]
    p_i = np.float32([tl, tr, bl, br])
    p_f = np.float32([[0,0],[width, 0], [0, height], [width, height]])
    M = cv.getPerspectiveTransform(p_i, p_f)
    
    warp = cv.warpPerspective(res, M, (width, height))
    if SHOW_WARP_WINDOW:
        cv.imshow("PERSPECTIVE WARP", warp)
    #/Perspective Transform
    
    #Binarization:
    warp_b = cv.threshold(warp, 10, 255, cv.THRESH_BINARY)[1]
    if SHOW_WARP_BINARY:
        cv.imshow("BINARY WARPED IMAGE", warp_b)
    #/Binarization
    
    return warp_b
#/PROCESS_IMAGE

#DETECT_LANES
def detect_lanes(img):
    #The purpose of this function is to calculate the position of lane lines based on a binary, top-down image of the road
    #
    #The main steps for this function are:
    #1. Quantization
    #2. Peak detection
    #3. Grouping
    #Modified sliding window:
    #/Sliding window
    pass
#/DETECT_LANES
###/FUNCTIONS

###HELPER FUNCTIONS
#CAMERA_SETUP
def camera_setup():
    cap = cv.VideoCapture(CAMERA_SOURCE, cv.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        print("Unable to open camera. Source: ", CAMERA_SOURCE)
        quit()
    return cap
#/CAMERA_SETUP

#TRACKBAR_SETUP
def trackbar_setup():
    cv.namedWindow("Trackbars")
    if USER_DEFINES_HLS:
        cv.createTrackbar("LH", "Trackbars", 90, 255, trackbar_call)
        cv.createTrackbar("UH", "Trackbars", 120, 255, trackbar_call)
        cv.createTrackbar("LL", "Trackbars", 60, 255, trackbar_call)
        cv.createTrackbar("UL", "Trackbars", 180, 255, trackbar_call)
        cv.createTrackbar("LS", "Trackbars", 40, 255, trackbar_call)
        cv.createTrackbar("US", "Trackbars", 160, 255, trackbar_call)
    if USER_DEFINES_PERSPECTIVE:
        cv.createTrackbar("PS", "Trackbars", 0, 1000, trackbar_call)
        cv.createTrackbar("PSH", "Trackbars", 0, 1000, trackbar_call)
    return
    
#/TRACKBAR_SETUP
    
#GPIO_SETUP
def gpio_setup():
    gpio.setmode(gpio.BCM)
#/GPIO_SETUP

#TRACKBAR_CALL
def trackbar_call(x):
    pass
#/TRACKBAR_CALL
###/HELPER FUNCTIONS

##########################################################
# Code for runtime implementation below this point
##########################################################

#####SETUP
cap = camera_setup()
trackbar_setup()


#####/SETUP

#####OPERATING LOOP
while True:
    #Operation loop outline:
    # 1. Get a frame from the camera
    # 2. Process image
    # 3. Detect lane lines from processed image
    # 4. Generate condensed lane image
    # 5. Give condensed lane image to neural network to obtain direction
    # 6. Update drive direction
	
	_, img = cap.read()
    
	img = process_image(img)
    #img = detect_lanes(img)
    
	k = cv.waitKey(5) & 0xFF
	if k == 27:
		break
cap.release()
#####/OPERATING LOOP