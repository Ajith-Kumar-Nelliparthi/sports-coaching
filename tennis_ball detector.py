import cv2 as cv
import numpy as np

img = cv.imread('images/a.jpg')
# cv.imshow('Original', img)

# rescale the image
def rescaleFrame(frame, scale=0.30):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_img = rescaleFrame(img)
# cv.imshow('Resized', resized_img)

# Convert RGB to HSV color space
hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV', hsv)

# defining lower and upper bounds for yellow color
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# create a mask using the bounds
mask = cv.inRange(hsv, lower_yellow, upper_yellow)
# cv.imshow('Mask', mask)

# find contours in the mask
contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# draw boundary box around the largest contour
if contours:
    # for single identification of the ball, we can use the largest contour
    """largest = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(largest)
    cv.rectangle(resized_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.putText(resized_img, "Ball", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2) """ 

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(resized_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv.putText(resized_img, "Ball", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    

# Display
cv.imshow('Tennis Ball Detection', resized_img)
cv.waitKey(0)
cv.destroyAllWindows()