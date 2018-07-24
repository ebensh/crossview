#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt

img_left = cv2.imread('drink_one.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('drink_two.jpg', cv2.IMREAD_GRAYSCALE)

img_left = cv2.GaussianBlur(img_left, (9, 9), 0)
img_right = cv2.GaussianBlur(img_right, (9, 9), 0)

#img_left = cv2.Canny(img_left, 200, 250)
#img_right = cv2.Canny(img_right, 200, 250)

#print(img_left.shape)
#print(img_right.shape)
#img_combined = cv2.subtract(img_left, img_right) 
#plt.imshow(img_combined)
#plt.show()
#exit(1)


# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp_left, des_left = orb.detectAndCompute(img_left,None)
kp_right, des_right = orb.detectAndCompute(img_right,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des_left, des_right)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x: x.distance)
# Draw first 10 matches.
img_combined = np.zeros(img_left.shape)
img_combined = cv2.drawMatches(img_left, kp_left,
                               img_right, kp_right,
                               matches[:10], img_combined, flags=2)

plt.imshow(img_combined)
plt.show()