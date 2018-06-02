# -*- coding: utf-8 -*-
"""
Tools for saving frame from the ZED camera

    Each frame needs to contains the followings:
        - RGB image
        - Depth image
        - Noirmal image
        - camera orientation
        - camera position in space

RGB, depth are just grabbed from the sdk
orientation and position needs to be initialized with an iOS devixce to set initial position (GPS) and initial orientation (acceletometer)
Needs a readme file to save camera settings and initial position and orientation


ReADME:
    scene description (camera direction, light settings, position and direction)
    camera settings
    date and time
    position of pixels to set white balance
    
    Camera direction
    
    
also wold be nice to have a position and orientation for each frame, to save in one file txt or whatever
"""

import cv2
import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl
import os

def main():
    
    zedcam = zcam.PyZEDCamera()
    init_params = zcam.PyInitParameters() 
    
    
    
    
    init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_QUALITY
    init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD720
   
  # Use a right-handed Y-up coordinate system
    init_params.coordinate_system = sl.PyCOORDINATE_SYSTEM.PyCOORDINATE_SYSTEM_RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.PyUNIT.PyUNIT_METER  # Set units in meters
    
    err = zedcam.open(init_params)
    if err != tp.PyERROR_CODE.PySUCCESS:
        exit(1)
    else: 
        print("Camera initialized")
        
    rgbImg = core.PyMat()
    normImg = core.PyMat()
    depthImg = core.PyMat()

    runtime_parameters = zcam.PyRuntimeParameters()
    
    

    

    sessionID = input("Enter session ID: ")
    cwd = os.getcwd()
    sampleDirPath = cwd + "\\" + sessionID + "\\samples\\"
    if not os.path.exists(sampleDirPath):
        os.makedirs(sampleDirPath)
    else:
        print("Directory with same session ID already exist!")
        exit(1)
    

    
    
    key = ''
    frameCount = 0
    while key != 113:  # for 'q' key
        err = zedcam.grab(runtime_parameters)
        if err == tp.PyERROR_CODE.PySUCCESS:
            zedcam.retrieve_image(rgbImg, sl.PyVIEW.PyVIEW_LEFT)
            zedcam.retrieve_image(depthImg, sl.PyVIEW.PyVIEW_DEPTH)
            zedcam.retrieve_measure(normImg, sl.PyMEASURE.PyMEASURE_NORMALS)
            cv2.imshow("ZED RGB", rgbImg.get_data())
            cv2.imshow("ZED Depth", depthImg.get_data())
            cv2.imshow("ZED Normals", normImg.get_data())
            rgbString = sampleDirPath + "RGB_" + str(frameCount) + ".png"
            depthString = sampleDirPath + "Depth_" + str(frameCount) + ".png"
            normString = sampleDirPath + "Norm_" + str(frameCount) + ".png"
            cv2.imwrite(rgbString, rgbImg.get_data())
            cv2.imwrite(depthString, depthImg.get_data())
            cv2.imwrite(normString, normImg.get_data())
            frameCount += 1;
            key = cv2.waitKey(5)
        else:
            key = cv2.waitKey(5)
    
    
    cv2.destroyAllWindows()
    readmePath = cwd + "\\" + sessionID + "\\readMe.txt"
    readmeFile = open(readmePath, "w");
    readmeFile.write("SessionID: " + str(sessionID))
    readmeFile.write("\nFPS: " + str(zedcam.get_camera_fps()))
    readmeFile.write("\nResolution: " + str(init_params.camera_resolution))
    readmeFile.write("\nCoord System: " + str(init_params.coordinate_system))
    readmeFile.write("\nUnit: " + str(init_params.coordinate_units))
    readmeFile.close()
    
    print("closing")
  # Close the camera
    zedcam.close()

def compute_ambient_occlusion(depthImg, shadowMaks):
    # per tutti i pixel in shadowMask calcolo il valore di ambient occlusion 
    return "stocazzo!"






if __name__ == "__main__":
    main()
