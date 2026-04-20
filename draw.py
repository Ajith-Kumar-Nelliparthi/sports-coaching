import cv2 as cv
import numpy as np

img = cv.imread('images/a.jpg')
cv.imshow('Original', img)

cv.waitKey(0)
cv.destroyAllWindows()