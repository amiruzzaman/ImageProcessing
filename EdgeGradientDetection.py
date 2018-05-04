# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 08:05:23 2017

@author: mamiruzz
"""

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import cv2

fileType = '.jpg'
numberStartRange = 10000000
numberEndRange = 90000000

originalDirectory = "C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\CurrentWork\\raw\\"
savedDirectory ="C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\CurrentWork\\processed\\"
#
#img = cv2.imread("C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\CurrentWork\\raw\\car-001.jpg", cv2.IMREAD_GRAYSCALE)
##img=mpimg.imread("C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\CurrentWork\\raw\\car-001.jpg")
#plt.imshow(img)
###############
## Laplacian
###############
#laplacian = cv2.Laplacian(img, cv2.CV_64F)
#cv2.imwrite('test.jpg', laplacian)
###############
## Sobel
###############
#sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 7) # ksize will be to be odd number
#sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 7) # ksize will be to be odd number
##imgplot=plt.imshow(sobely)
#
###############
## Canny
###############
#th, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#edgesCanny = cv2.Canny(img, th/2, th)
##imgplot=plt.imshow(edgesCanny)
##plt.show()


##############
# 2D sharpening/averaging 
##############
import numpy as np
#kernel = np.ones((5,5),np.float32)/25
#averaging = cv2.filter2D(img,-1,kernel)
#
#plt.subplot(121),plt.imshow(img),plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(averaging),plt.title('Averaging')
#plt.xticks([]), plt.yticks([])
#plt.show()



   
from os import listdir
from os.path import isfile, join
import numpy as np

##############
# Get first 3 letters
##############
def FirstThree(s):
    return s[:3]


##############
# Laplacian
##############    
def LaplacianImage(imagePath, savePath, imageName, randNumber):
    laplacian = cv2.Laplacian(imagePath, cv2.CV_64F)
    newPath = savePath + FirstThree(imageName)+str(randNumber) + fileType
    cv2.imwrite(newPath, laplacian)
    print("Laplacian image processed!\n")


##############
# Sobelx
##############
def SobelXImage(imagePath, savePath, imageName, randNumber):
    sobelx = cv2.Sobel(imagePath, cv2.CV_64F, 1, 0, ksize = 7) # ksize will be to be odd number
    newPath = savePath + FirstThree(imageName)+str(randNumber) + fileType
    cv2.imwrite(newPath, sobelx)
    print("Sobelx image processed!\n")

##############
# Sobely
##############    
def SobelYImage(imagePath, savePath, imageName, randNumber):
    sobely = cv2.Sobel(imagePath, cv2.CV_64F, 0, 1, ksize = 7) # ksize will be to be odd number
    newPath = savePath + FirstThree(imageName)+str(randNumber) + fileType
    cv2.imwrite(newPath, sobely)
    print("Sobely image processed!\n")

##############
# Canny
##############  
def CannyImage(imagePath, savePath, imageName, randNumber):
    th, bw = cv2.threshold(imagePath, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edgesCanny = cv2.Canny(imagePath, th/2, th)
    newPath = savePath + FirstThree(imageName)+str(randNumber) + fileType
    cv2.imwrite(newPath, edgesCanny)
    print("Canny image processed!\n")
    
##############
# Image sharpenning/averaging 
##############  
def AveragingImage(imagePath, savePath, imageName, randNumber):
    kernel = np.ones((5,5),np.float32)/25 # kernel will be to be odd number
    averaging = cv2.filter2D(imagePath,-1,kernel)
    newPath = savePath + FirstThree(imageName)+str(randNumber) + fileType
    cv2.imwrite(newPath, averaging)
    print("Averaging image processed!\n")
    

from PIL import Image
from resizeimage import resizeimage

def ResizeFile(in_file, out_file, size):
    basewidth = size
    img = Image.open(in_file)
    wpercent = (basewidth/(float(img.size[0])))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(out_file, format="JPEG") 
    #print("Image resize complete!")

def PreProcessImage(orgDirectory, svDirectory):
    files = [f for f in listdir(orgDirectory) if isfile(join(orgDirectory, f))]
    #print (len(files))
    
    i = 0
    while i <len(files):
        fullPath = orgDirectory + files[i]
        img = cv2.imread(fullPath, cv2.IMREAD_GRAYSCALE)
        #check what you see
        #plt.imshow(img)
        
        ##############
        # Laplacian
        ##############
        
        from random import randint
        randomNumber = randint(numberStartRange, numberEndRange)
        LaplacianImage(img, svDirectory, files[i], randomNumber)
        
        ##############
        # Sobelx
        ##############
        
        randomNumber = randint(numberStartRange, numberEndRange)
        SobelXImage(img, svDirectory, files[i], randomNumber)
        
        ##############
        # Sobely
        ##############
        randomNumber = randint(numberStartRange, numberEndRange)
        SobelYImage(img, svDirectory, files[i], randomNumber)
        
        
        ##############
        # Canny
        ##############
        randomNumber = randint(numberStartRange, numberEndRange)
        CannyImage(img, svDirectory, files[i], randomNumber)
                
        
        ##############
        # 2D sharpenning/averaging 
        ##############
        randomNumber = randint(numberStartRange, numberEndRange)
        AveragingImage(img, svDirectory, files[i], randomNumber)
       
        #print(newPath)
        i+= 1

#PreProcessImage(originalDirectory, savedDirectory)

#savedDirectory = 'C:\\Users\\mamiruzz\\Downloads\\Netbeans\\Python\\kaggledata\\test\\'
#savedDirectory = 'C:\\Users\\mamiruzz\\Downloads\\Netbeans\\Python\\MulticlassClassification\\src\\ASLtrainingResize\\a'
savedDirectory = 'C:\\Users\\mamiruzz\\Downloads\\HouseDataMix\\train\\Bad\\'


#imageSize = 32
imageSize = 128

import os   
def ResizeDirectoryFiles(savedDirectory, imageSize):
    files = os.listdir(savedDirectory)
    j = 0
    total = 0;
    while j< len(files):
        f = files[j]
        j = j+1
        ResizeFile(savedDirectory +f, savedDirectory +"bad"+f, imageSize)
        total = total + j
    print(total)

def RenameDirectoryFiles(savedDirectory, startLetter):
    #test = 'C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\HouseData\\test'
    for dpath, dnames, fnames in os.walk(savedDirectory):
        i = 0
        for f in fnames:
            os.chdir(dpath)
            if f.startswith(startLetter):
                #os.rename(f, f.replace('n', 'cat.'))
                os.rename(f, startLetter+'.' + str(i) + '.jpg')
                i = i + 1
    #            print(i)
    #            print('\\n')
    print('Rename is done!')

#RenameDirectoryFiles(savedDirectory, 'a')
ResizeDirectoryFiles(savedDirectory, imageSize)
#from random import randint
#randomNumber = randint(numberStartRange, numberEndRange)
#print(randomNumber)




