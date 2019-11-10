
import cv2 as cv
import numpy as np
import numpy.polynomial.polynomial as poly
import cardriver as car
from time import sleep

###CONSTANTS
CAMERA_SOURCE = 0

IMG_RESCALE = 4

SHOW_PROCESSING_WINDOWS = False
SHOW_RESULT_WINDOW = False
HARDWARE_ENABLE = True
PRINT_DIRECTION = False
RECORD_VIDEO = True
###/CONSTANTS

###SETUP FUNCTIONS
def cameraSetup():
    cap = cv.VideoCapture(CAMERA_SOURCE)
    if cap is None or not cap.isOpened():
        print("Unable to open camera. Source: ", CAMERA_SOURCE)
        quit()
    return cap
    
def recordingSetup():
    out = cv.VideoWriter("out.avi", cv.VideoWriter_fourcc(*"H264"), 30, (160, 120))
    return out
###/SETUP FUNCTIONS

###RUNTIME FUNCTIONS
#COLOUR_FILTER
def colourFilter(img):
    #Function overview:
    #   1. Convert to HLS colourspace
    #   2. Filter out specified colour range
    #   4. Return black/white filtered image
    
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    lowerHLSThreshold = np.array([0, 0, 0])
    upperHLSThreshold = np.array([255, 161, 112])
    hlsFiltered = cv.inRange(hls, lowerHLSThreshold, upperHLSThreshold)

    return hlsFiltered
#/COLOUR_FILTER

#PERSPECTIVE_TRANSFORM
def perspectiveTransform(img):
    #Function overview:
    #   1. Expand image width to reduce losses from original image after transform
    #   2. Compute transform homography
    #   3. Apply transform
    #   4. Return transformed image
    
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    
    #Construct the indexes of a trapezoid as the homography:
    widthScale = 0.22
    heightScale = 0.6
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
    iPT = cv.getPerspectiveTransform(finalPoints, initialPoints)
    warp = cv.warpPerspective(img, M, (2*imgWidth, imgHeight))
    
    return warp, iPT
#/PERSPECTIVE_TRANSFORM

#DETECT_CENTROIDS
def detectCentroids(img):
    #Function overview:
    #   1. Divide the input image into n horizontal slices
    #   2. Iterate over those slices, determine the locations of the centroids within the slice
    #   3. Return the coordinates of the centroids in each slice
    numSections = 10
    contourThresh = 50
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    
    lanePoints = []
    for i in range(numSections):
        roi = img[i * imgHeight / numSections : (i + 1)*imgHeight / numSections, 0:imgWidth]
        im2, contours, hierarchy = cv.findContours(roi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv.moments(c)
            if M["m00"] != 0 and M["m00"] > contourThresh:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                lanePoints.append([cX, cY + i * imgHeight / numSections])
    return lanePoints
#/DETECT_CENTROIDS

#CLASSIFY_LANE_POINTS
def classifyLanePoints(lanePoints):    
    #Function overview:
    #   1. Sort points based on x value
    #   2. Group points based on x value using a threshold
    #   3. Return grouped points
    #If only one lane is present, it will be passed as the left lane
    leftLane = []
    rightLane = []
    groupingDistanceThreshold = 50
    sortedLanePoints = sorted(lanePoints, key=lambda x: x[0])

    if len(sortedLanePoints) > 1:
        while len(sortedLanePoints) > 0:
            #Two neighboring points will be considered part of the same lane if their x vales are close together
            #Example: x values = 1, 2, 3, 10, 11, 12
            #First pass: [1, 2], 3, 10, 11, 12. [1, 2] same line, pop 1 to leftLane
            #Second pass: [2, 3], 10, 11, 12. [2, 3] same line, pop 2 to leftLane
            #Third pass: [3, 10], 11, 12. [3, 10] not same line, pop 3 to leftLane, remaining go to rightLane
            if abs(sortedLanePoints[1][0] - sortedLanePoints[0][0]) < groupingDistanceThreshold:
                #Points on same line, pop to leftLane, continue
                leftLane.append(sortedLanePoints.pop(0))
                if len(sortedLanePoints) == 1:
                    #If all points are on the same line must pop again 
                    leftLane.append(sortedLanePoints.pop(0))
                    break
            else:
                #Points not on same line, pop one to leftLane, rest are rightLane
                leftLane.append(sortedLanePoints.pop(0))
                break
    rightLane = sortedLanePoints
	
    return leftLane, rightLane
#/CLASSIFY_LANE_POINTS

#FIT_LANES
def fitLanes(leftLanePoints, rightLanePoints):
    #Function overview:
    #1. Fit straight line to points in left and right lanes
    #2. Return coefficients for these lines
    leftCoefs = []
    rightCoefs = []
    if len(leftLanePoints) > 1:
        x = [p[0] for p in leftLanePoints]
        y = [p[1] for p in leftLanePoints]
        leftCoefs = poly.polyfit(y, x, 1)
    if len(rightLanePoints) > 1:
        x = [p[0] for p in rightLanePoints]
        y = [p[1] for p in rightLanePoints]
        rightCoefs = poly.polyfit(y, x, 1)
    return leftCoefs, rightCoefs
#/FIT_LANES

#GET_AVG_COEFS
def getAvgCoefs(leftCoefs, rightCoefs):
    if len(leftCoefs) == 0 and len(rightCoefs) == 0:
        return []
    if len(leftCoefs) == 0:
        return rightCoefs
    if len(rightCoefs) == 0:
        return leftCoefs
    return np.mean([leftCoefs, rightCoefs], axis=0)
#/GET_AVG_COEFS

#SHOW_LANE_POINTS
def showLanePoints(img, lanePoints):
    result = img.copy()
    for point in lanePoints:
        cv.circle(result, (point[0], point[1]), 5, (0, 0, 0), -1)
    cv.imshow("Lane Points", result)
#/SHOW_LANE_POINTS

#SHOW_CLASSIFIED_LANE_POINTS
def showClassifiedLanePoints(img, leftLanePoints, rightLanePoints):
    result = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    if len(leftLanePoints) > 0:
        for point in leftLanePoints:
            cv.circle(result, (point[0], point[1]), 5, (255, 0, 0), -1)
    if len(rightLanePoints) > 0:
        for point in rightLanePoints:
            cv.circle(result, (point[0], point[1]), 5, (0, 0, 255), -1)
    cv.imshow("Classified Lane Points", result)
    return
#/SHOW_CLASSIFIED_LANE_POINTS

#SHOW_LANE_LINES
def showLaneLines(img, leftCoefs, rightCoefs, avgCoefs):
    result = cv.cvtColor(img.copy(), cv.COLOR_GRAY2BGR)
    x = [0, result.shape[0]]
    if len(leftCoefs) > 0:
        fit = poly.polyval(x, leftCoefs)
        cv.line(result, (int(fit[0]), x[0]), (int(fit[1]), x[1]), (255, 0, 0), 5)
        cv.putText(result, str(leftCoefs[1])[0:5], (0, 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv.LINE_AA)
    if len(rightCoefs) > 0:
        fit = poly.polyval(x, rightCoefs)
        cv.line(result, (int(fit[0]), x[0]), (int(fit[1]), x[1]), (0, 0, 255), 5)
        cv.putText(result, str(rightCoefs[1])[0:5], (0, 40), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv.LINE_AA)
    if len(avgCoefs) > 0:
        fit = poly.polyval(x, avgCoefs)
        cv.line(result, (int(fit[0]), x[0]), (int(fit[1]), x[1]), (255, 0, 255), 5)
        cv.putText(result, str(avgCoefs[1])[0:5], (0, 60), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1, cv.LINE_AA)
    cv.imshow("Lane Lines", result)
#/SHOW_LANE_LINES

#COMPUTE_RESULTS
def computeResults(img, direction, warpedShape, leftCoefs, rightCoefs, avgCoefs, iPT):
    warpedLines = np.zeros((warpedShape[0], warpedShape[1], 3), np.uint8)
    x = [0, warpedLines.shape[1]]
    if len(leftCoefs) > 0:
        fit = poly.polyval(x, leftCoefs)
        cv.line(warpedLines, (int(fit[0]), x[0]), (int(fit[1]), x[1]), (255, 0, 0), 5)
    if len(rightCoefs) > 0:
        fit = poly.polyval(x, rightCoefs)
        cv.line(warpedLines, (int(fit[0]), x[0]), (int(fit[1]), x[1]), (0, 0, 255), 5)
    if len(avgCoefs) > 0:
        fit = poly.polyval(x, avgCoefs)
        cv.line(warpedLines, (int(fit[0]), x[0]), (int(fit[1]), x[1]), (255, 0, 255), 5)
    unwarpedLines = cv.warpPerspective(warpedLines, iPT, (warpedLines.shape[1], warpedLines.shape[0]))
    unwarpedLines = unwarpedLines[0:img.shape[0], 0:img.shape[1]]
    
    result = cv.addWeighted(img, 0.8, unwarpedLines, 1, 0)
    
    if len(leftCoefs) > 0:
        cv.putText(result, str(leftCoefs[1])[0:5], (0, 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv.LINE_AA)
    if len(rightCoefs) > 0:
        cv.putText(result, str(rightCoefs[1])[0:5], (0, 40), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv.LINE_AA)
    if len(avgCoefs) > 0:
        cv.putText(result, str(avgCoefs[1])[0:5], (0, 60), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1, cv.LINE_AA)
        
    if direction == 0:
        cv.putText(result, "Stop", (img.shape[1] - int(img.shape[1]/2), 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
    elif direction == 1:
        cv.putText(result, "Forward", (img.shape[1] - int(img.shape[1]/2), 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
    elif direction == 2:
        cv.putText(result, "Left", (img.shape[1] - int(img.shape[1]/2), 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
    else:
        cv.putText(result, "Right", (img.shape[1] - int(img.shape[1]/2), 20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
    return result
#/COMPUTE_RESULTS

#COMPUTE_DRIVE_DIRECTION
def computeDriveDirection(avgSlope):
    leftThreshold = 0.15
    rightThreshold = -0.15
    
    if len(avgSlope) == 0:
        return 0
    
    slope = avgSlope[1]
    if slope < leftThreshold and slope > rightThreshold:
        return 1
    elif slope >= leftThreshold:
        return 2
    else:
        return 3
#/COMPUTE_DRIVE_DIRECTION

#DRIVE
def drive(direction):
    if direction == 0:
        if PRINT_DIRECTION:
            print("Stop.")
        if HARDWARE_ENABLE:
            car.stop()
    elif direction == 1:
        if PRINT_DIRECTION:
            print("Forward.")
        if HARDWARE_ENABLE:
            car.forward()
    elif direction == 2:
        if PRINT_DIRECTION:
            print("Left.")
        if HARDWARE_ENABLE:
            car.left()
    else:
        if PRINT_DIRECTION:
            print("right")
        if HARDWARE_ENABLE:
            car.right()
    return
#/DRIVE
###/RUNTIME FUNCTIONS

###MAIN LOOP:
def main():
    print("Program started.")
    cap = cameraSetup()
    out = recordingSetup()
    while True:
        #Loop overview:
        #   1. Read input from camera
        #   2. Resize image
        #   3. Filter out lane lines
        #   4. Perform perspective transform
        #   5. Perform centroid detection on horizontal slices of transformed image
        #   6. Group centroid points into lanes
        #   7. Perform curve-fitting to lane points
        #   8. Compute driving direction based on lane curves
        #   9. Determine hardware output based on driving direction, drive
        
        #1. Read input from camera
        _, frame = cap.read()
        
        #2. Resize image to improve processing times
        resizedFrame = cv.resize(frame, (int(frame.shape[1] / IMG_RESCALE), int(frame.shape[0] / IMG_RESCALE)), interpolation = cv.INTER_AREA)
        if SHOW_PROCESSING_WINDOWS:
            cv.imshow("Resized Frame", resizedFrame)
        
        #3. Filter out lane lines
        filteredFrame = colourFilter(resizedFrame)
        if SHOW_PROCESSING_WINDOWS:
			cv.imshow("Filtered Frame", filteredFrame)
        
        #4. Perform perspective transform
        warpedFrame, iPT = perspectiveTransform(filteredFrame) #obtain inverse perspective transform
        if SHOW_PROCESSING_WINDOWS:
			cv.imshow("Warped Frame", warpedFrame)
        
        #5. Perform centroid detection on horizontal slices of transformed image
        lanePoints = detectCentroids(warpedFrame)
        if SHOW_PROCESSING_WINDOWS:
			showLanePoints(warpedFrame, lanePoints)

        #6. Group centroid points into lanes
        leftLanePoints, rightLanePoints = classifyLanePoints(lanePoints)
        if SHOW_PROCESSING_WINDOWS:
			showClassifiedLanePoints(warpedFrame, leftLanePoints, rightLanePoints)
        
        #7. Perform curve-fitting to lane points, normalize slopes
        leftCoefs, rightCoefs = fitLanes(leftLanePoints, rightLanePoints)
        avgCoefs = getAvgCoefs(leftCoefs, rightCoefs)
        if SHOW_PROCESSING_WINDOWS:
            showLaneLines(warpedFrame, leftCoefs, rightCoefs, avgCoefs)
            
        #8. Compute driving direction based on average slope of lane lines
        direction = computeDriveDirection(avgCoefs)
        
        #9. Display computational results
        result = computeResults(resizedFrame, direction, warpedFrame.shape, leftCoefs, rightCoefs, avgCoefs, iPT)
        out.write(result)
        if SHOW_RESULT_WINDOW:
            cv.imshow("Result", result)
            
        #9. Determine hardware output based on driving direction
        drive(direction)
        
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            cap.release()
            out.release()
            cv.destroyAllWindows()
            print("Program exiting.")
            quit()
main()
###/MAIN LOOP
