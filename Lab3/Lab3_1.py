import cv2
import numpy as np

# Load the image and convert it to grayscale
image_path = 'Mount1.png'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate image derivatives Ix and Iy
Ix = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
Iy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

# Calculate measures IxIx, IyIy, and IxIy
IxIx = Ix * Ix
IyIy = Iy * Iy
IxIy = Ix * Iy

# Calculate structure matrix components as weighted sum of nearby measures
window_size = 5
S_IxIx = cv2.GaussianBlur(IxIx, (window_size, window_size), 0)
S_IyIy = cv2.GaussianBlur(IyIy, (window_size, window_size), 0)
S_IxIy = cv2.GaussianBlur(IxIy, (window_size, window_size), 0)

# Calculate Harris "cornerness" as an estimate of 2nd eigenvalue: det(S) / tr(S)
det_S = S_IxIx * S_IyIy - S_IxIy * S_IxIy
trace_S = S_IxIx + S_IyIy
cornerness = det_S / trace_S

# Set a cornerness threshold
threshold = 100
corners = np.where(cornerness > threshold)

# Run non-max suppression on the response map
w = 10
cornerness = cv2.dilate(cornerness, np.ones((w, w)))

# Get the final corner points
final_corners = np.where(cornerness == cornerness[corners])

# Plot the corner points on the image
for y, x in zip(final_corners[0], final_corners[1]):
    cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), (0, 255, 0), 1)

# Display the image with detected corners
cv2.imshow('Harris Corner Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
