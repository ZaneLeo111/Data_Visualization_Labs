import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image and extract a row/column
# img1 = plt.imread("dogsmall.jpg")
# signal = img1[0, :]  # extract the first row

# print(signal)

img = cv.imread('dogsmall.jpg')
row_pixels = img[2, :]
# row_pixels = img[0 ,2, :]
# cv.imshow("row", row_pixels)

print(row_pixels)

# cv.waitKey(0)
# cv.destroyAllWindows()

# sampling rate
sr = 128
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

# print(t)