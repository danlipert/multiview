import extrinsic
import numpy as np
import cv2

img1 = cv2.imread('left.jpg',0)   # queryimage # left image
img2 = cv2.imread('right.jpg',0) #trainimage    # right image
img1 = cv2.resize(img1, (0,0), fx=0.6, fy=0.6)
img2 = cv2.resize(img2, (0,0), fx=0.6, fy=0.6)

print 'resizing done... finding fundamental matrix'

F = extrinsic.fundamentalMatrixFromImages(img1, img2)

print 'fundamental matrix found... calculating essential matrix'

#intrinsic camera matrix
A = [[595.451828822481, 0.0, 248.40265456322769],[0.0,550.0701337760904,133.03834795529602],[0.0,0.0,1.0]]

#calculate essential matrix
E = np.transpose(A) * F * A
print 'essential matrix:'
print E

print 'determinant check:'
det = np.linalg.det(E)
print det






