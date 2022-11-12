# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:56:18 2022

@author: ndhee
"""
import cv2
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from typing import Sized
from PIL import Image
import math
import random
import time

currDir=os.getcwd()+"/testPatient"
filePath=currDir
slicesPath=os.getcwd()+"/Slices"

def checkFolderAndFileExist(folderName,imageName):
    
    if not os.path.isdir(folderName):
        os.mkdir(folderName)
    if not os.path.isdir(os.path.join(folderName,imageName)):
        os.mkdir(os.path.join(folderName,imageName))
        
def fixingHeighWidthToSliceImages(contours, imageName,filePath):
    o=0
    i=0
    image = cv2.imread(os.path.join(filePath,imageName))
    a1, b1, w1, h1 = cv2.boundingRect(contours[len(contours)-1])
    while(i<len(contours)):
        a2, b2, w1, h1 = cv2.boundingRect(contours[len(contours)-i-1])
        if(b2 != b1 and abs(b2-b1) > h1):
            break
        i+=1
    i=0    
    while(i<len(contours)):    
        a3, b3, w1, h1 = cv2.boundingRect(contours[len(contours)-i-1])
        if(a3 != a1 and abs(a3-a1) > w1):
            break
        i+=1 
    slicedImageWidth =  a3 - a1 
    slicedImageHeight = b2 - b1
    sliceImageIntoFrames(contours,imageName,filePath,slicedImageWidth,slicedImageHeight,a1,b1)
    cluster()
    print(o)
    o+=1
    
def checkThreshInImageName():
    if os.path.isdir(slicesPath):
        shutil.rmtree(slicesPath)
    for FMRI in os.listdir(filePath):
        if 'thresh' in FMRI:
            convertToGrayToDetectR(filePath, FMRI)
            

    
def convertToGrayToDetectR(filePath,imageName):
    
    checkFolderAndFileExist(slicesPath,imageName.split('.')[0])
    image = cv2.imread(os.path.join(filePath,imageName)) 
    BGR2GRAY=cv2.Canny(cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),255,255),255, 255) 
    contours, hierarchy = cv2.findContours(BGR2GRAY.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    fixingHeighWidthToSliceImages(contours, imageName,filePath)
    
def checkingAndDrawingBackgroundIfImageExisit(a1,b1,a2,b2,image,imageId,imageName):     
    BGR2GRAY=cv2.Canny(cv2.inRange(cv2.cvtColor(image[b2:b1, a1:a2],cv2.COLOR_BGR2GRAY),0,25), 50, 100) 
    contour, hierarchy = cv2.findContours(BGR2GRAY, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    if(len(contour)):
        cv2.imwrite(os.path.join(slicesPath,imageName.split('.')[0])+'/'+'sliceID '+str(imageId-3) +'.png', image[b2:b1, a1:a2])
    
    

def sliceImageIntoFrames(contours,imageName,filePath,slicedImageWidth,slicedImageHeight,Max_x,Max_y):
    imageId=0
    image = cv2.imread(os.path.join(filePath,imageName)) 
    for i in range(len(contours)):
        contours_contours = contours[len(contours)-i-1]
        a1, b1, w1, h1 = cv2.boundingRect(contours_contours)
        a2 = a1+slicedImageWidth-w1
        b2 = b1-slicedImageHeight
        a1 +=2 * w1
        b1 +=2 * h1
        if not (a1 < Max_x or b2 < Max_y): 
            checkingAndDrawingBackgroundIfImageExisit(a1,b1,a2,b2,image,imageId,imageName)
            imageId+= 1

            
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    v = mx
    H = h / 2
    S = s * 255.0
    V = v * 255.0
    return H, S, V
def red(H, S, V):
    if(((H>=0 and H<=10) or (H>=156 and H<=180)) and S>=43 and S<=255 and V>=46 and V<=255):
        return True

def org(H, S, V):
    if(H>=11 and H<=25 and S>=43 and S<=255 and V>=46 and V<=255):
        return True

def yellow(H, S, V):
    if(H>=26 and H<=34 and S>=43 and S<=255 and V>=46 and V<=255):
        return True
def blue(H, S, V):
    if(H>=100 and H<=124 and S>=43 and S<=255 and V>=46 and V<=255):
        
        return True
def cluster():    
    k=0
    d_l=os.listdir(slicesPath)
    for  k in d_l:
        folder_path=slicesPath+'/'+k
        for i in os.listdir(folder_path):
            L_path=folder_path+'/'+i
            print(L_path)
            L_image=Image.open(L_path)
            out = L_image.convert("RGB")
            dataset=np.array(out)
            size = dataset.shape
            count = 0
            dataset2=[[0,0]]
            for i in range(size[0]):
                for j in range(size[1]):
                    temp0,temp1,temp2 = rgb2hsv(dataset[i,j,0],dataset[i,j,1],dataset[i,j,2])
                    if(red(temp0,temp1,temp2)):
                        count = count + 1 
                        dataset[i,j,0] = 255
                        dataset[i,j,1] = 255
                        dataset[i,j,2] = 0
                        coords=tuple([i,j])
                        dataset2.append(coords)
                    elif (org(temp0,temp1,temp2)):
                        count = count + 1 
                        dataset[i,j,0] = 255
                        dataset[i,j,1] = 255
                        dataset[i,j,2] = 0
                        coords=tuple([i,j])
                        dataset2.append(coords)
                    elif (yellow(temp0,temp1,temp2)):
                        count = count + 1 
                        dataset[i,j,0] = 255
                        dataset[i,j,1] = 255
                        dataset[i,j,2] = 0
                        coords=tuple([i,j])
                        dataset2.append(coords)
                    elif (blue(temp0,temp1,temp2)):
                        count = count + 1 
                        dataset[i,j,0] = 255
                        dataset[i,j,1] = 255
                        dataset[i,j,2] = 0
                        coords=tuple([i,j])
                        dataset2.append(coords)
                    else:
                        dataset[i,j,0] = 255
                        dataset[i,j,1] = 255
                        dataset[i,j,2] = 255
        
            plt.imshow(dataset)
            gray = cv2.cvtColor(dataset, cv2.COLOR_BGR2GRAY)
            db = DBSCAN(eps=2.5,min_samples=5)
            clustering=db.fit(dataset2)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters=0
        
            data_pd = pd.DataFrame(pd.value_counts(labels))
            data_pd.columns=['date']
            for index, row in data_pd.iterrows():
              if(row['date']>135):
                n_clusters+=1
            print(data_pd)
            print("Estimated number of clusters: %d" % n_clusters+"%d" % k)
            k+=1

        

     
    
    
    
    
    
    
    