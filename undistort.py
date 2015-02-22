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
    calibration_path = sys.argv[2]
except Exception as e:
    print 'Please enter the calibration json file path'
    exit()

#load calibration data
json_data = json.load(open(calibration_path))
dist = np.array(json_data['dist'])
camera_mtx = np.array(json_data['mtx'])

#load video
#load video
print 'Loading video...'
video = cv2.VideoCapture(video_path)
total_frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
lastgray = None

samples_to_use = 50
frame_count = 0

while(video.isOpened()):
    frame_count = frame_count + 1
    print 'Reading frame %s of %s' % (frame_count, total_frames)
    
    flag, img = video.read()
    h, w, d = img.shape
    undist_img = cv2.undistort(img, camera_mtx, dist)
    cv2.imshow('img',undist_img)
    cv2.waitKey(250)
    
    
    
    
    
    
