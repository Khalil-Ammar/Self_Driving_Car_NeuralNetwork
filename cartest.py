import cardriver as car
import cv2 as cv

cv.namedWindow("Car Test")
while True:
	k = cv.waitKey(5) & 0xFF
	if k == 27:
		quit()
	elif k == 119:
		car.forward()
	elif k == 97:
		car.left()
	elif k == 100:
		car.right()
	elif k == 115:
		car.backward()
	else:
		car.stop() 
