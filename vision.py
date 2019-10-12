import cv2 as cv 
import numpy as np

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
_, frame = cap.read()
HEIGHT, WIDTH, CHANNELS = frame.shape
STEP = 5
XWIDTH = 60
SCALEX = 10
SCALEY = 10

while(1):
	_, frame = cap.read()
	cv.imshow("frame", frame)
	
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	cv.imshow("hsv", hsv)
	
	# define range of blue color in HSV
	lw = np.array([150,150,150])
	uw = np.array([255,255,255])
	mask = cv.inRange(frame, lw, uw)
	res = cv.bitwise_and(frame, frame, mask=mask)
	cv.imshow("res", res)
	
	uwidth = 0.225
	lwidth = 0.025
	# TL, TR, BR, BL
	tl = [uwidth*WIDTH, 0.5*HEIGHT]
	tr = [(1-uwidth)*WIDTH, 0.5*HEIGHT]
	br = [(1-lwidth)*WIDTH, HEIGHT]
	bl = [lwidth*WIDTH, HEIGHT]
	p_i = np.float32([tl, tr, bl, br])
	p_f = np.float32([[0,0], [WIDTH, 0], [0, HEIGHT], [WIDTH, HEIGHT]])
	M = cv.getPerspectiveTransform(p_i, p_f)
	
	warp = cv.warpPerspective(res, M, (WIDTH, HEIGHT))
	warp_gray = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
	warp_bw =  cv.threshold(warp_gray, 150, 255, cv.THRESH_BINARY)[1]
	
	roadpts = []
	for y in range(SCALEY - 1, -1, -1):
		for x in range(int(WIDTH - XWIDTH)):
			x1 = WIDTH - (x + XWIDTH)
			y1 = int(HEIGHT * y / SCALEY)
			x2 = WIDTH - x
			y2 = int(HEIGHT * (y+1)/SCALEY)
			roi = warp_bw[y1:y2, x1:x2]
			avg = np.average(roi)
			if (avg > 210):
				warp = cv.rectangle(warp, (x1, y1), (x2, y2), (255, 255, 0), 2)
				roadpts.append([x1 + int((x2-x1)/2), y1 + int((y2-y1)/2)])
				break
	
	for y in range(SCALEY):
		for x in range(int(WIDTH - XWIDTH)):
			
			x1 = x
			y1 = int(HEIGHT * y / SCALEY)
			x2 = x + XWIDTH
			y2 = int(HEIGHT * (y+1)/SCALEY)
			roi = warp_bw[y1:y2, x1:x2]
			avg = np.average(roi)
			if (avg > 210):
				warp = cv.rectangle(warp, (x1, y1), (x2, y2), (0, 255, 255), 2)
				roadpts.append([x1 + int((x2-x1)/2), y1 + int((y2-y1)/2)])
				break
				
	
	
	cv.fillPoly(warp, np.array([roadpts]), (0, 255, 255))
	cv.imshow("warp_bw", warp_bw)
	cv.imshow("warp", warp)
	k = cv.waitKey(5) & 0xFF
	if k == 27:
		break

cap.release()
cv.destroyAllWindows()