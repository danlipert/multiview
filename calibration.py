import numpy as np
import cv2
import sys

#parse arguments
try:
    video_path = sys.argv[1]
except Exception as e:
    print 'Please enter the calibration video path'
    exit()

try:
    output_path = sys.argv[2]
except Exception as e:
    output_path = './camera.json'
    


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#load video
video = cv2.VideoCapture(video_path)
lastgray = None
while(video.isOpened()):
    flag, img = video.read()

    try:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    except Exception as e:
        break
    lastgray = gray
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print 'found chessboard'
        objpoints.append(objp)

        #corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #imgpoints.append(corners2)
        imgpoints.append(corners)
        print 'samples: %i' % len(imgpoints)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(1)

print 'calibrating...'
gray = lastgray
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print 'calibration complete'
print ret, mtx, dist, rvecs, tvecs

mean_error = 0
tot_error = 0

for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", mean_error/len(objpoints)

cv2.destroyAllWindows()



