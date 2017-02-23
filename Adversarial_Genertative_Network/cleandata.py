# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:39:37 2017

@author: rbarnes

The goal of this script is to prepare images for analysis.
"""

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

class Cleaner:
    """Make sure that the path specified has a trailing slash"""
    
    def __init__(self,path,batch_size):
        if path[-1]!='/' or path[-2:]!='\\':
            path=path+'\\'
        self.path=path
        print(path)
        self.batch_size=batch_size
        
    def clean_photos(self):
        idg1=ImageDataGenerator()
        idg=ImageDataGenerator(
                               horizontal_flip=True,
                               vertical_flip=True,
         #                      zca_whitening=True
                               )
        #idg.fit(idg1.flow_from_directory(self.path,batch_size=2500,class_mode=None,target_size=(64,64)).next())
        print(self.path)
        return idg.flow_from_directory(self.path,batch_size=self.batch_size,class_mode=None,target_size=(64,64))
        
        
    
if __name__=='__main__':
    images=Cleaner('h:\\Desktop\\GitHub\\practice_deep_learning\\deep_learning\\Adversarial_Genertative_Network\\Data\\',1).clean_photos()
    z=np.mean(images.next(),axis=0).reshape((1,64,64,3))
    Image.fromarray(z[0,:,:,:].astype(np.uint8)).save('h:\\desktop\\github\\practice_deep_learning\\deep_learning\\Adversarial_Genertative_Network\\Results\\testimgsize.png')