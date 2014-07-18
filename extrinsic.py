import numpy as np
import cv2
import sys
import json

def calculateExtrinsic(img1, img2, cameraIntrinsic):
    """
    Takes 2 images, plus intrinsic parameters and returns extrinsic parameters between the two photos
    Assumes same camera
    """
    sift = cv2.SIFT()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
 
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
 
    flann = cv2.FlannBasedMatcher(index_params,search_params)
 
    matches = flann.knnMatch(des1,des2,k=2)
 
    good = []
    pts1 = []
    pts2 = []
 
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts2 = np.float32(pts2)
    pts1 = np.float32(pts1)       
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    return F

def fundamentalMatrixFromImages(img1, img2):
    sift = cv2.SIFT()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
 
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
 
    flann = cv2.FlannBasedMatcher(index_params,search_params)
 
    matches = flann.knnMatch(des1,des2,k=2)
 
    good = []
    pts1 = []
    pts2 = []
 
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts2 = np.float32(pts2)
    pts1 = np.float32(pts1)       
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    return F


