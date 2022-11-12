import numpy as np
import cv2
import os
import csv
import shutil
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import Image

def check_dir(file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save(file_name, records):
    check_dir(file_name)
    headerList = ['slice number', 'count']
    with open(file_name,'w+') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(headerList)
    with open(file_name,'a+') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in records.items():
            writer.writerow([key, value])

def clusterimages(threshpath, clusterspath,path):
    images=os.listdir(threshpath)
    if not os.path.exists(clusterspath):
        os.makedirs(clusterspath)
    for image in images:
        threshname=threshpath.split('/')[-1]
        clusterpath= os.path.join(clusterspath, threshname)
        if not os.path.exists(clusterpath):
            os.makedirs(clusterpath)
    mylist={}

    for image in images:
        imagepath= os.path.join(threshpath,image)
        threshname=threshpath.split('/')[-1]
        clusterpath= os.path.join(clusterspath, threshname)

        val=clustermyimage(imagepath, clusterpath)
        idx=imagepath.split('/')[-1].index('slice.jpg')
        mylist[imagepath.split('/')[-1][0:idx]]=val
        save(clusterpath+"/"+threshname+".csv",mylist)

    
def clustermyimage(imagepath, savepath):
    
    path=imagepath
    img = cv2.imread(path) 
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(grayscale, (0, 25, 40), (25, 255, 255))
    mask2 = cv2.inRange(grayscale, (110, 50, 50), (130, 255, 255))
    mask = mask1 + mask2
    res = cv2.bitwise_and(img, img, mask=mask)
    c = 0
    l1 = len(mask)
    if mask[0] is not None:
        l2 = len(mask[0])
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == 0:
                c = c+1
    # print(c)

    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)

    if(abs(c-l1*l2)<=100):
        return 0
    else:
        im_pil = Image.fromarray(mask)
        im_np = np.asarray(im_pil)
        # print(imagepath.split('/')[-1])
        cv2.imwrite(os.path.join(savepath , imagepath.split('/')[-1]),im_np)
        img = res
        feature_image = np.reshape(img, [-1, 3])
        rows, cols, chs = img.shape

        db = DBSCAN(eps=0.215, min_samples=135, metric='euclidean', algorithm='auto')
        db.fit(feature_image)
        labels = db.labels_
        clustercount = len(set(labels)) - (1 if -1 in labels else 0)
        # print(clustercount)
        return clustercount
        # plt.figure(2)
        # plt.subplot(2, 1, 1)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.subplot(2, 1, 2)
        # plt.imshow(np.reshape(labels, [rows, cols]))
        # plt.axis('off')
        # plt.show()
    
