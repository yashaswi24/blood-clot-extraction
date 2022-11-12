from turtle import pen
import numpy as np
from PIL import Image
import cv2
import matplotlib as plot
import os
import shutil
from brainextraction import *
from clustering import *



# Assuming the following files/folders in same folder
# brainextraction.py
# clustering.py
# test.py
# template.png
# testPatient


if __name__ == '__main__':
    # print("ok")
    folder="testPatient"
    path= os.getcwd()
    # print(path)
    testpath = os.path.join(path, folder)


    boundarypath=os.path.join(path,"boundaries")
    slicespath=os.path.join(path,"slices")
    clusterspath= os.path.join(path,"clusters")
    if os.path.exists(boundarypath):
        shutil.rmtree(boundarypath)
    os.makedirs(boundarypath)
    if os.path.exists(slicespath):
        shutil.rmtree(slicespath)
    os.makedirs(slicespath)
    if os.path.exists(clusterspath):
        shutil.rmtree(clusterspath)
    os.makedirs(clusterspath)



    templatepath= os.path.join(path,'template.png')
    print("template path ",templatepath)
    for images in os.listdir(testpath):
        if images.endswith(".png") and images.split("_")[-1]== "thresh.png" :
            imgpath=os.path.join(testpath,images)
            image = cv2.imread(imgpath)
            idx=images.split("_")[1]
            template = cv2.imread(templatepath,0)
            
            lists=matchtemplate(image,template) 
            try:
                w, h = template.shape[::-1]
                savecroppedimagesslices(image,idx,lists,slicespath,w,h)

                # drawcontour(image)
                # savecroppedimagesboundaries(image,idx,lists,boundarypath,w,h)
            except:
                print("Template image not in same folder as python files")
    
    slicesfolder=os.listdir(slicespath)
    for thresh in slicesfolder:
        # print(thresh)
        threshfolder= os.path.join(slicespath,thresh)
        clusterimages(threshfolder,clusterspath,path)


    # for images in os.listdir(testpath):
    #     if images.endswith(".png") and images.split("_")[-1]== "thresh.png" :



 
    

    
