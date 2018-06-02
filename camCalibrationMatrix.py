# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:46:37 2018

@author: Fulvio Bertolini
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
  
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

def multiplyMatrices4x4(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):
    
  
    
    rvec_iPhone[1,0] = rvec_iPhone[1,0] * -1
    rvec_ZED[1,0] = rvec_ZED[1,0] * -1
    tvec_iPhone[1] = tvec_iPhone[1] * -1
    tvec_ZED[1] = tvec_ZED[1] * -1
    rotM_iPhone = cv2.Rodrigues(rvec_iPhone)[0]
    rotM_ZED = cv2.Rodrigues(rvec_ZED)[0]
    
    
    iPhone_2_ChB = np.zeros((4,4), np.float)
    ZED_2_ChB = np.zeros((4,4), np.float)
    
    iPhone_2_ChB[0:3,0:3] = rotM_iPhone
    ZED_2_ChB[0:3,0:3] = rotM_ZED
    
    iPhone_2_ChB[0,3] = tvec_iPhone[0]
    iPhone_2_ChB[1,3] = tvec_iPhone[1]
    iPhone_2_ChB[2,3] = tvec_iPhone[2]
    ZED_2_ChB[0,3] = tvec_ZED[0]
    ZED_2_ChB[1,3] = tvec_ZED[1]
    ZED_2_ChB[2,3] = tvec_ZED[2]
    
    iPhone_2_ChB[3,3] = 1
    ZED_2_ChB[3,3] = 1
    
    ChB_2_ZED = np.linalg.inv(ZED_2_ChB)
    ChB_2_iPhone = np.linalg.inv(iPhone_2_ChB)
    iPhone_2_ZED = np.matmul(iPhone_2_ChB, ChB_2_ZED)
    
    ZED_2_iPhone = np.matmul(ZED_2_ChB, ChB_2_iPhone)
    
    
    tvec = iPhone_2_ZED[0:3,3]
    rvec = cv2.Rodrigues(iPhone_2_ZED[0:3,0:3])[0]
    

    origin = np.zeros((4,1), np.float)
    origin[3] = 1
    print("checkerboard origin in checkerboard coord (according to iPhone_2Chk): ", np.matmul(ChB_2_iPhone, origin))
    iPhone2ZED:np.ndarray((4,4), np.float32) = np.matmul(ZED_2_ChB, ChB_2_iPhone)
    iPhone2ZED_inv = np.linalg.inv(iPhone2ZED)
    print("\nzed transf\ntvec: ", np.matmul(iPhone2ZED, origin))
    print("rvec:\n ", cv2.Rodrigues(iPhone2ZED[0:3,0:3])[0])
    return rvec, tvec, iPhone2ZED_inv
  
def computeARKit_2_ZED_matrix(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED):

    rvec, tvec, matrix = multiplyMatrices4x4(rvec_iPhone, tvec_iPhone, rvec_ZED, tvec_ZED)
    
    
    return rvec, tvec, matrix
    
    

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
            cv2.waitKey(10)
    
    cv2.destroyAllWindows()
    
    
    return (rvecs, tvecs)


def getCameraIntrinsic(images):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp = np.multiply(objp, 0.0246)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
            #cv2.namedWindow("img", cv2.WINDOW_NORMAL )        # Create window with freedom of dimensions
            #cv2.resizeWindow("img", 800, 600)              # Resize window to specified dimensions
    
            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (9,6), corners,ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)
    
    #cv2.destroyAllWindows()
    
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    
    
   
    
    return (mtx, dist, rvecs, tvecs)


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


#images_ZED = [ "./calibrationDebugImages/ZED/A/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/B/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/C_30/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/C_45/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/D_-30/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/D_-45/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/E/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/F_15/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/F_30/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/G_-10/samples/RGB_0.PNG",
#                  "./calibrationDebugImages/ZED/G_-20/samples/RGB_0.PNG" ]
#


#
#images_iPhone = ["./calibrationDebugImages/iPhone/A.JPG",
#					"./calibrationDebugImages/iPhone/B.JPG",
#					"./calibrationDebugImages/iPhone/C_30.JPG",
#					"./calibrationDebugImages/iPhone/C_45.JPG",
#					"./calibrationDebugImages/iPhone/D_-30.JPG",
#					"./calibrationDebugImages/iPhone/D_-45.JPG",
#					"./calibrationDebugImages/iPhone/E.JPG",
#					"./calibrationDebugImages/iPhone/F_15.JPG",
#					"./calibrationDebugImages/iPhone/F_30.JPG",
#					"./calibrationDebugImages/iPhone/G_-10.JPG",
#					"./calibrationDebugImages/iPhone/G_-20.JPG" ]


images_ZED = ['./intrinsic_ZED/rgb_1019.png',
                 './intrinsic_ZED/rgb_103.png',
                 './intrinsic_ZED/rgb_104.png',
                 './intrinsic_ZED/rgb_1052.png',
                 './intrinsic_ZED/rgb_1055.png',
                 './intrinsic_ZED/rgb_1058.png',
                 './intrinsic_ZED/rgb_1066.png',
                 './intrinsic_ZED/rgb_111.png',
                 './intrinsic_ZED/rgb_125.png',
                 './intrinsic_ZED/rgb_137.png',
                 './intrinsic_ZED/rgb_152.png',
                 './intrinsic_ZED/rgb_159.png',
                 './intrinsic_ZED/rgb_162.png',
                 './intrinsic_ZED/rgb_163.png',
                 './intrinsic_ZED/rgb_166.png',
                 './intrinsic_ZED/rgb_168.png',
                 './intrinsic_ZED/rgb_206.png',
                 './intrinsic_ZED/rgb_230.png',
                 './intrinsic_ZED/rgb_254.png',
                 './intrinsic_ZED/rgb_255.png',
                 './intrinsic_ZED/rgb_259.png',
                 './intrinsic_ZED/rgb_264.png',
                 './intrinsic_ZED/rgb_267.png',
                 './intrinsic_ZED/rgb_269.png',
                 './intrinsic_ZED/rgb_270.png',
                 './intrinsic_ZED/rgb_273.png',
                 './intrinsic_ZED/rgb_277.png',
                 './intrinsic_ZED/rgb_283.png',
                 './intrinsic_ZED/rgb_290.png',
                 './intrinsic_ZED/rgb_291.png',
                 './intrinsic_ZED/rgb_296.png',
                 './intrinsic_ZED/rgb_310.png',
                 './intrinsic_ZED/rgb_313.png',
                 './intrinsic_ZED/rgb_323.png',
                 './intrinsic_ZED/rgb_328.png',
                 './intrinsic_ZED/rgb_330.png',
                 './intrinsic_ZED/rgb_331.png',
                 './intrinsic_ZED/rgb_342.png',
                 './intrinsic_ZED/rgb_343.png',
                 './intrinsic_ZED/rgb_357.png',
                 './intrinsic_ZED/rgb_358.png',
                 './intrinsic_ZED/rgb_371.png',
                 './intrinsic_ZED/rgb_373.png',
                 './intrinsic_ZED/rgb_405.png',
                 './intrinsic_ZED/rgb_447.png',
                 './intrinsic_ZED/rgb_459.png',
                 './intrinsic_ZED/rgb_470.png',
                 './intrinsic_ZED/rgb_471.png',
                 './intrinsic_ZED/rgb_482.png',
                 './intrinsic_ZED/rgb_494.png',
                 './intrinsic_ZED/rgb_498.png',
                 './intrinsic_ZED/rgb_501.png',
                 './intrinsic_ZED/rgb_504.png',
                 './intrinsic_ZED/rgb_511.png',
                 './intrinsic_ZED/rgb_533.png',
                 './intrinsic_ZED/rgb_548.png',
                 './intrinsic_ZED/rgb_560.png',
                 './intrinsic_ZED/rgb_561.png',
                 './intrinsic_ZED/rgb_562.png',
                 './intrinsic_ZED/rgb_566.png',
                 './intrinsic_ZED/rgb_570.png',
                 './intrinsic_ZED/rgb_574.png',
                 './intrinsic_ZED/rgb_575.png',
                 './intrinsic_ZED/rgb_581.png',
                 './intrinsic_ZED/rgb_592.png',
                 './intrinsic_ZED/rgb_615.png',
                 './intrinsic_ZED/rgb_621.png',
                 './intrinsic_ZED/rgb_623.png',
                 './intrinsic_ZED/rgb_628.png',
                 './intrinsic_ZED/rgb_630.png',
                 './intrinsic_ZED/rgb_631.png',
                 './intrinsic_ZED/rgb_654.png',
                 './intrinsic_ZED/rgb_661.png',
                 './intrinsic_ZED/rgb_663.png',
                 './intrinsic_ZED/rgb_666.png',
                 './intrinsic_ZED/rgb_670.png',
                 './intrinsic_ZED/rgb_675.png',
                 './intrinsic_ZED/rgb_678.png',
                 './intrinsic_ZED/rgb_680.png',
                 './intrinsic_ZED/rgb_683.png',
                 './intrinsic_ZED/rgb_689.png',
                 './intrinsic_ZED/rgb_692.png',
                 './intrinsic_ZED/rgb_705.png',
                 './intrinsic_ZED/rgb_710.png',
                 './intrinsic_ZED/rgb_712.png',
                 './intrinsic_ZED/rgb_714.png',
                 './intrinsic_ZED/rgb_719.png',
                 './intrinsic_ZED/rgb_726.png',
                 './intrinsic_ZED/rgb_732.png',
                 './intrinsic_ZED/rgb_737.png',
                 './intrinsic_ZED/rgb_775.png',
                 './intrinsic_ZED/rgb_790.png',
                 './intrinsic_ZED/rgb_793.png',
                 './intrinsic_ZED/rgb_800.png',
                 './intrinsic_ZED/rgb_807.png',
                 './intrinsic_ZED/rgb_811.png',
                 './intrinsic_ZED/rgb_820.png',
                 './intrinsic_ZED/rgb_823.png',
                 './intrinsic_ZED/rgb_832.png',
                 './intrinsic_ZED/rgb_838.png',
                 './intrinsic_ZED/rgb_841.png',
                 './intrinsic_ZED/rgb_849.png',
                 './intrinsic_ZED/rgb_860.png',
                 './intrinsic_ZED/rgb_862.png',
                 './intrinsic_ZED/rgb_884.png',
                 './intrinsic_ZED/rgb_887.png',
                 './intrinsic_ZED/rgb_891.png',
                 './intrinsic_ZED/rgb_917.png',
                 './intrinsic_ZED/rgb_923.png',
                 './intrinsic_ZED/rgb_965.png',
                 './intrinsic_ZED/rgb_968.png',
                 './intrinsic_ZED/rgb_97.png',
                 './intrinsic_ZED/rgb_970.png',
                 './intrinsic_ZED/rgb_980.png',
                 './intrinsic_ZED/rgb_982.png',
                 './intrinsic_ZED/rgb_986.png']

images_iPhone = ["./intrinsic_iPhone_screenshots/IMG_2185.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2186.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2187.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2188.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2189.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2190.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2191.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2192.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2193.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2194.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2195.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2196.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2197.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2198.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2199.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2200.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2201.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2202.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2203.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2204.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2205.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2206.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2207.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2208.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2209.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2210.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2211.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2212.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2213.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2214.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2215.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2216.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2217.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2218.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2219.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2220.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2221.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2222.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2223.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2224.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2225.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2226.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2227.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2228.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2229.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2230.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2231.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2232.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2233.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2234.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2235.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2236.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2237.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2238.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2239.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2240.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2241.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2242.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2243.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2244.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2245.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2246.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2247.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2248.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2249.PNG",
				 "./intrinsic_iPhone_screenshots/IMG_2250.PNG"]


(mtx, dist, rvecs, tvecs) = getCameraIntrinsic(images_ZED)

#rvecs_ZED, tvecs_ZED = getCameraExtrinsic(images_ZED, cameraMatrix_ZED, distCoeff_ZED)
#rvecs_iPhone, tvecs_iPhone = getCameraExtrinsic(images_iPhone, cameraMatrix_iPhone, distCoeff_iPhone)
#
#originZED = np.zeros((len(images_ZED),4,1), np.float)
#origin_iPhone = np.zeros((len(images_iPhone),4,1), np.float)
#originZED[:,3,0] = 1
#origin_iPhone[:,3,0] = 1
#
#pointsZED = np.zeros((len(images_ZED),4,1), np.float)
#points_iPhone = np.zeros((len(images_iPhone),4,1), np.float)
#rotationsCkb_2_ZED = np.zeros((len(images_ZED),3,1), np.float)
#rotationsCkb_2_iPhone = np.zeros((len(images_iPhone),3,1), np.float)
#rotationsZED_2_Ckb = np.zeros((len(images_ZED),3,1), np.float)
#rotations_iPhone_2_Ckb = np.zeros((len(images_iPhone),3,1), np.float)
#ZEDCam_2_Chkb = np.zeros((len(images_ZED),4,4), np.float)
#iPhoneCam_2_Chkb = np.zeros((len(images_iPhone),4,4), np.float)
#Chkb_2_ZEDCam = np.zeros((len(images_ZED),4,4), np.float)
#Chkb_2_iPhoneCam = np.zeros((len(images_iPhone),4,4), np.float)
#
#
#for i in range(0,11):
#    ZEDCam_2_Chkb[i,:,:], Chkb_2_ZEDCam[i,:,:] = getCameraMatrix(rvecs_ZED[i], tvecs_ZED[i])
#    iPhoneCam_2_Chkb[i,:,:], Chkb_2_iPhoneCam[i,:,:] = getCameraMatrix(rvecs_iPhone[i], tvecs_iPhone[i])
#    
#    pointsZED[i,:,:] = np.matmul(ZEDCam_2_Chkb[i,:,:], originZED[i,:,:])
#    originZED[i,:,:] = np.matmul(Chkb_2_ZEDCam[i,:,:], pointsZED[i,:,:])
#    points_iPhone[i,:,:] = np.matmul(iPhoneCam_2_Chkb[i,:,:], origin_iPhone[i,:,:])
#    origin_iPhone[i,:,:] = np.matmul(Chkb_2_iPhoneCam[i,:,:], points_iPhone[i,:,:])
#    
#    rotationsZED_2_Ckb[i,:,:] = cv2.Rodrigues(Chkb_2_ZEDCam[i,0:3, 0:3])[0]
#    rotationsCkb_2_ZED[i,:,:] = cv2.Rodrigues(ZEDCam_2_Chkb[i,0:3, 0:3])[0]
#    rotations_iPhone_2_Ckb[i,:,:] = cv2.Rodrigues(Chkb_2_iPhoneCam[i,0:3, 0:3])[0]
#    rotationsCkb_2_iPhone[i,:,:] = cv2.Rodrigues(iPhoneCam_2_Chkb[i,0:3, 0:3])[0]
#
#
#
#iPhoneCam_2_ZEDCam = np.matmul(Chkb_2_ZEDCam[5,:,:], iPhoneCam_2_Chkb[0,:,:])
#
#
#
#matrixFile = open("../TCP/ARKitCam_2_ZEDCam.txt", "a+")
#matrixFile.write(writeMatrix(iPhoneCam_2_ZEDCam) + "\n")
#matrixFile.close()
#
#print("ah scemooooo")
#
#
