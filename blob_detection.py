# Standard imports
import cv2
import numpy as np
import scipy.misc
# Read image
im = cv2.imread("blob2.png", cv2.IMREAD_GRAYSCALE)
 
# Set up the detector with default parameters.
orb = cv2.ORB_create(nfeatures=110, edgeThreshold=20)
 
# Detect blobs.
kp1, des1 = orb.detectAndCompute(im, None)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, kp1, np.array([]), (0,0,255))
scipy.misc.imsave('outfile.jpg', im_with_keypoints)
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)