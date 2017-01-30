# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:39:37 2017

@author: rbarnes

The goal of this script is to prepare images for analysis.
"""

import cv2
import os
import numpy as np

class Cleaner:
    """Make sure that the path specified has a trailing slash"""
    
    def __init__(self,path):
        if path[-1]!='/' or path[-2:]!='\\':
            path=path+'\\'
        self.path=path
        self.photos=os.listdir(self.path)
        
    
    def clean_photo(self,photo):
        img=cv2.imread(self.path+photo,0)
        return(img[int(img.shape[0]/2)-56:int(img.shape[0]/2)+56,int(img.shape[1]/2)-56:int(img.shape[1]/2)+56])
        
    def build_X(self):
        X=[]
        for obj in self.photos:
            temp=self.clean_photo(obj)
            if (temp.shape[0]==112 and temp.shape[1]==112):
                X.append(temp.reshape((1,)+temp.shape))
        return np.array(X)
        
        
    
if __name__=='__main__':
    print(Cleaner('h:\\Desktop\\GitHub\\practice_deep_learning\\deep_learning\\Adversarial_Genertative_Network\\Data\\train').build_X())