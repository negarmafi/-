import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pl
image = cv2.imread('C:/Users/Negin/Downloads/Documents/naghshe.JPEG')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
plt.plot(hist_hue)
plt.title('Histogram of Hue Channel')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')
plt.show()

image = cv2.imread('C:/Users/Negin/Downloads/Documents/naghshe.JPEG')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_bound = np.array([150,150,0])
upper_bound = np.array([255,255, 100])
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
segmented_image = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()