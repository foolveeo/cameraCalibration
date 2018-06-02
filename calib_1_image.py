# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:38:50 2018

@author: Fulvio Bertolini
"""


import numpy as np
import cv2
import os
  
def getCameraExtrinsic(images, mtx, dist):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp = np.multiply(objp, 0.0246)
    
   
   
    tvecs = []
    rvecs = [] 
    
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            _, rvec, tvec, inliers = cv2.solvePnPRansac(objp,corners2, mtx, dist)
            rvecs.append(rvec)
            tvecs.append(tvec)
            
            cv2.namedWindow("img", cv2.WINDOW_NORMAL )        # Create window with freedom of dimensions
            cv2.resizeWindow("img", 2400, 1200)              # Resize window to specified dimensions
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners,ret)
            cv2.imshow('img',img)
            cv2.waitKey(2000)
    
    cv2.destroyAllWindows()
    
    
    return (rvecs, tvecs)

def getCameraMatrix(rvec, tvec):
    Cam_2_Chkb = np.zeros((4,4), np.float)
    Chkb_2_Cam = np.zeros((4,4), np.float)
    
    
    rotM_Chkb_2_Cam = cv2.Rodrigues(rvec)[0]
    
    Chkb_2_Cam[0:3, 0:3] = rotM_Chkb_2_Cam
    Chkb_2_Cam[0:3, 3] = tvec.ravel()
    Chkb_2_Cam[3,3] = 1
    
    Cam_2_Chkb = np.linalg.inv(Chkb_2_Cam)
    
    
    return Cam_2_Chkb, Chkb_2_Cam
    
    
    
    
    
    
def writeMatrix(matrix):
    
    string:str = ""
    for i in range(0,matrix.shape[0]):
        for j in range(0,matrix.shape[1]):
            string += "{:.9f}".format(matrix[i,j])
            if(j != 3):
                string += " "
        if(i != 3):
                string += "\t"
    
    return string

image_ZED = ["../Sessions/carnevale3/ZED/checkerBoard.png"]
image_iPhone = ["../Sessions/carnevale3/iPhone/checkerBoard.png"]


# zed intrinsic parameters
cameraMatrix_ZED = np.zeros((3,3), np.float64)
cameraMatrix_ZED[0,0] = 660.445
cameraMatrix_ZED[0,2] = 650.905
cameraMatrix_ZED[1,1] = 660.579
cameraMatrix_ZED[1,2] = 327.352
cameraMatrix_ZED[2,2] = 1

distCoeff_ZED = np.zeros((1,5), np.float64)
distCoeff_ZED[0,0] = -0.00174692
distCoeff_ZED[0,1] = -0.0174969	
distCoeff_ZED[0,2] = -0.000182398
distCoeff_ZED[0,3] = -0.00751098
distCoeff_ZED[0,4] =  0.0219687

# iphone intrinsic parameters
cameraMatrix_iPhone = np.zeros((3,3), np.float64)
cameraMatrix_iPhone[0,0] = 3513.23
cameraMatrix_iPhone[0,2] = 1542.31
cameraMatrix_iPhone[1,1] = 3519.98
cameraMatrix_iPhone[1,2] = 2089.89
cameraMatrix_iPhone[2,2] = 1

distCoeff_iPhone = np.zeros((1,5), np.float64)
distCoeff_iPhone[0,0] = 0.293461	
distCoeff_iPhone[0,1] = -1.3054	
distCoeff_iPhone[0,2] = 0.0132138	
distCoeff_iPhone[0,3] = 0.00104743
distCoeff_iPhone[0,4] = 2.50754


rvecs_ZED, tvecs_ZED = getCameraExtrinsic(image_ZED, cameraMatrix_ZED, distCoeff_ZED)
rvecs_iPhone, tvecs_iPhone = getCameraExtrinsic(image_iPhone, cameraMatrix_iPhone, distCoeff_iPhone)

ZEDCam_2_Chkb = np.zeros((4,4), np.float)
iPhoneCam_2_Chkb = np.zeros((4,4), np.float)
Chkb_2_ZEDCam = np.zeros((4,4), np.float)
Chkb_2_iPhoneCam = np.zeros((4,4), np.float)


ZEDCam_2_Chkb[:,:], Chkb_2_ZEDCam[:,:] = getCameraMatrix(rvecs_ZED[0], tvecs_ZED[0])
iPhoneCam_2_Chkb[:,:], Chkb_2_iPhoneCam[:,:] = getCameraMatrix(rvecs_iPhone[0], tvecs_iPhone[0])

iPhoneCam_2_ZEDCam = np.matmul(Chkb_2_ZEDCam[:,:], iPhoneCam_2_Chkb[:,:])



matrixFile = open("../TCP/ARKitCam_2_ZEDCam.txt", "a+")
matrixFile.write(writeMatrix(iPhoneCam_2_ZEDCam) + "\n")
matrixFile.close()
print(iPhoneCam_2_ZEDCam)
print("ah scemooooo")
   