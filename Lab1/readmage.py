import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
img = cv2.imread('dogsmall.jpg')

# Convert the image to a 2D array with pixel locations and RGB values
height, width, channels = img.shape
pixels = []
for y in range(height):
    row = []
    for x in range(width):
        pixel = img[y,x]
        row.append((x, y, pixel[2], pixel[1], pixel[0]))
    pixels.append(row)
pixels = np.array(pixels)

# Plot the image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
ax.set_title('Original Image')

# Plot the 2D array with RGB values
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('RGB Values')
for row in pixels:
    for pixel in row:
        ax.scatter(pixel[0], pixel[1], color=[pixel[2]/255, pixel[3]/255, pixel[4]/255], s=5)
plt.show()
