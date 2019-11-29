import imutils
import cv2
import numpy as np
import time

## downsample frame
def getPyramids(image, scale = 1.5, minSize = 30, maxSize = 1000):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width = w)
        if(image.shape[0] < minSize or image.shape[1] < minSize):
            break
        if (image.shape[0] > maxSize or image.shape[1] > maxSize):
            continue
        yield image

## sliding window to compute MSE
def getSearchWindows(image, step, windowSize):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

## compute MSE
def computeMSE(img, refImg):
    assert img.shape == refImg.shape, "Images must be the same shape. img shape = {0}, imgRef shape = {1}".format(img.shape, refImg.shape)
    error = np.sum((img.astype("float") - refImg.astype("float")) ** 2)
    error = error/float(img.shape[0] * img.shape[1] * img.shape[2])
    return error

if __name__ == "__main__":
    ##get target image
    frame_img_path = "./Object_Detection/dataset/Stop Sign Dataset/3.jpg"
    frame = cv2.imread(frame_img_path)
    frame = imutils.resize(frame, width = 500)

    reference_path = "./target.png"
    reference = cv2.imread(reference_path)

    maxSim = -1
    currBox = (0,0,0,0)

    t0 = time.time()

    ##find best match
    for pyramid in getPyramids(reference, minSize = 50, maxSize=frame.shape[0]):
        for (x, y, window) in getSearchWindows(frame, 2, pyramid.shape):
            # print(pyramid.shape, window.shape)
            if window.shape[0] != pyramid.shape[0] or window.shape[1] != pyramid.shape[1]:
                continue
            sim = 1/computeMSE(pyramid, window)
            if sim > maxSim :
                maxSim = sim
                currBox = (x, y, pyramid.shape[0], pyramid.shape[1])

    ##draw box
    print("Execution time: {0}".format(time.time() - t0))
    (x, y, w, h) = currBox
    print(currBox)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
