# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:16:36 2017

@author: mamiruzz
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

RawDataFolder = 'C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\CurrentWork\\train\\'
RotatedDataFolder = 'C:\\Users\\mamiruzz\\Downloads\\bitbucket\\Python\\CurrentWork\\rotated\\'

def ImageRotation(RawDataFolder, RotatedDataFolder):
    files = os.listdir(RawDataFolder)
    
    j = 0
    while j< len(files):
        f = files[j]
        j = j+1
        img = load_img(RawDataFolder +f)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        pre = f[:3]
        #print(pre)
        SaveRotatedImage(x, 20, 1, pre, RotatedDataFolder)
      
        print("Image rotation is complete and files are saved in "+RotatedDataFolder)
        
        ResizeFile(RawDataFolder +f, RawDataFolder +f, 64)
                
def SaveRotatedImage(x, n, b, pre, rotateddata):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    i = 0
    for batch in datagen.flow(x, batch_size=b, save_to_dir=rotateddata, save_prefix=pre, save_format='jpg'):
        i += 1
        if i > n:
            break  # otherwise the generator would loop indefinitely


from PIL import Image
from resizeimage import resizeimage

def ResizeFile(in_file, out_file, size):
    basewidth = size
    img = Image.open(in_file)
    wpercent = (basewidth/(float(img.size[0])))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(out_file, format="JPEG") 
    
ImageRotation(RawDataFolder, RotatedDataFolder) 
            




print("working")
 











