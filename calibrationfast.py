import numpy as np
import cv2
import sys
import json

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

samples_to_use = 50

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
#use only a few samples for speeeed
objpoints = objpoints[:samples_to_use]
imgpoints = imgpoints[:samples_to_use]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print 'calibration complete'
print 'ret: %s' % ret
print 'mtx: %s' % mtx
print 'dist: %s' % dist
print 'rvecs: %s' % rvecs
print 'tvecs: %s' % tvecs

mean_error = 0
tot_error = 0

for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    print 'error: %f' % error
    tot_error = tot_error + error

mean_error = (float(tot_error)/float(samples_to_use))
print "total error: %f" %  mean_error

cv2.destroyAllWindows()

data = {'ret':ret, 'mtx':mtx, 'dist':dist, 'rvecs':rvecs, 'tvecs':tvecs, 'name':"iPhone 5", 'mean_error':mean_error}

datastring = json.dumps(data, sort_keys=True, indent=4, separators=(',', ': '), cls=NumpyAwareJSONEncoder)

print datastring

fp = open(output_path, 'w')
fp.write(datastring)
fp.close()





