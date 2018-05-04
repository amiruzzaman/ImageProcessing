# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:12:45 2017

@author: mamiruzz
"""



#import cv2
#camera = cv2.VideoCapture(0)
#while True:
#    return_value,image = camera.read()
#    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    cv2.imshow('image',gray)
#    if cv2.waitKey(1)& 0xFF == ord('s'):
#        cv2.imwrite('test.jpg',image)
#        break
#camera.release()
#cv2.destroyAllWindows()





#import cv2
#
#cpt = 0
#maxFrames = 1 # if you want 1 frames only.
#
#try:
#    vidStream = cv2.VideoCapture(0) # index of your camera
#except:
#    print ("problem opening input stream")
#    sys.exit(1)
#
#while cpt < maxFrames:
#    ret, frame = vidStream.read() # read frame and return code.
#    if not ret: # if return code is bad, abort.
#        sys.exit(0)
#    cv2.imshow("test window", frame) # show image in window
#    cv2.imwrite("image%04i.jpg" %cpt, frame)
#    cpt += 1


#import numpy as np
import cv2


#cap = cv2.VideoCapture(0)
#count = 0
#
#while count < 1:
#    # Capture frame-by-frame
#    ret, frame = cap.read()
#    
#    # Our operations on the frame come here
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    fileName = "test.%d.jpg"%count
#    cv2.imwrite(fileName, frame)     # save frame as JPEG file
#    count +=1
#    
#    # Display the resulting frame
#    cv2.imshow('test',gray)
#    if cv2.waitKey(10):
#        break
#cap.release();
#cv2.destroyAllWindows()






fileType = 'jpg'
numberOfImage = 1



def GetImageFromWebCam(imageName, numberOfImage):
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < numberOfImage:
       # Capture frame-by-frame
       ret, frame = cap.read()
    
       # Our operations on the frame come here
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       #gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
       fileName = imageName+".%d.jpg"%count
       cv2.imwrite(fileName, gray)
       #cv2.imwrite(fileName, frame)     # save frame as JPEG file
       count +=1
    
       # Display the resulting frame
       #cv2.imshow('test',gray)
#       if cv2.waitKey(10):
#          break
    cap.release();
    cv2.destroyAllWindows()

GetImageFromWebCam('test', numberOfImage)