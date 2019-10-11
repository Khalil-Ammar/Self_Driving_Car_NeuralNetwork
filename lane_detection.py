import cv2
import numpy as np
import time
import math

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('sliders')
cv2.createTrackbar('kernel', 'sliders', 2, 5, nothing)
cv2.createTrackbar('lowCanny', 'sliders', 50, 255, nothing)
cv2.createTrackbar('highCanny', 'sliders', 150, 255, nothing)
cv2.createTrackbar('minLineLength', 'sliders', 20, 640, nothing)
cv2.createTrackbar('maxLineGap', 'sliders', 10, 20, nothing)
cv2.createTrackbar('houghThreshold', 'sliders', 20, 200, nothing)
cv2.createTrackbar('dilate', 'sliders', 0, 5, nothing)

mask = cv2.imread('mask.jpg', 0)
avg_p_slope = 1
avg_p_int = 1
avg_n_slope = 1
avg_n_int = 1

while(1):
    kernel = 2*cv2.getTrackbarPos('kernel', 'sliders') + 1
    lowCanny = cv2.getTrackbarPos('lowCanny', 'sliders')
    highCanny = cv2.getTrackbarPos('highCanny', 'sliders')
    minLineLength = cv2.getTrackbarPos('minLineLength', 'sliders')
    maxLineGap = cv2.getTrackbarPos('maxLineGap', 'sliders')
    houghThreshold = cv2.getTrackbarPos('houghThreshold', 'sliders')
    dilate_size = cv2.getTrackbarPos('dilate', 'sliders')
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_size+1, 2*dilate_size+1), (dilate_size, dilate_size))

    _, frame = cap.read()

    height, width = frame.shape[:2]
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(grayscale, (kernel, kernel), 0)
    edges = cv2.Canny(gblur, lowCanny, highCanny)
    result = cv2.bitwise_and(edges, mask)
    result = cv2.dilate(result, element)
    
    if (0):
	    lines = cv2.HoughLinesP(result, 1, np.pi/180, houghThreshold, minLineLength, maxLineGap)
	    if lines is not None:
		for i in range(0, len(lines)):
			l = lines[i][0]
		cv2.line(frame, (l[0], l[1]), (l[2], l[3]), 3, cv2.LINE_AA)

    lines = cv2.HoughLines(result, 1, np.pi/180, houghThreshold, 0, 0)
    if lines is not None:
	x0 = []
	y0 = []
	for i in range(0, len(lines)):
		rho = lines[i][0][0]
		theta = lines[i][0][1]
		a = math.cos(theta)
		b = math.sin(theta)
		x0.append(a * rho)
		y0.append(b * rho)
	
	x0 = np.mean(x0)
	y0 = np.mean(y0)
	
	pt1 = (int(x0+640*(-b)), int(y0+640*(a)))
	pt2 = (int(x0-640*(-b)), int(y0-640*(a)))
	cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow('frame',frame)
    # cv2.imshow('gblur', gblur)
    # cv2.imshow('edges', edges)
    # cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()