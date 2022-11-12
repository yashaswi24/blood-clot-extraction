from json.tool import main
import numpy as np
from PIL import Image
import cv2
import matplotlib as plot
import os


def matchtemplate(image,template):
    w, h = template.shape[::-1]
    
    img_gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.8
    loc = np.where(res >= threshold)
    points=zip(*loc[::-1])
    mylist=[]
    for pt in points:
        mylist.append((pt[0] + w, pt[1] + h))

    return mylist
def drawcontour(image):
    print("Drawing contours")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))

    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

def saveimagetofolder(image,k,path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if hierarchy is not None:
        cv2.imwrite(os.path.join(path , str(k)+'slice.jpg'), image)

def findwidthheight(lists):

    x=lists[0][0]
    y=lists[0][1]
    width=0
    length=0
    for idx in range(len(lists)):
        if(idx!=0):
            val1=lists[idx][0]
            val2=lists[idx][1]
            if(val2==y and width==0):
                width=val1-x
            if(val1==x and length==0) :
                length=val2-y
    return width,length
def savecroppedimagesslices(image,imgname,array,parent_dir,tw,th):
    # print("saving images")
    id=findwidthheight(array)

    w=id[0]-tw
    h=id[1]-th
    
    directory= "ICthresh"+str(imgname)+"/"
    path = os.path.join(parent_dir, directory)
    exists = os.path.exists(path)
    if not exists:
        os.mkdir(path)

    for index,item in enumerate(array):

        i=item[0]
        j=item[1]
        cropped_image= image[j:j+h, i:i+w]
        cv2.imshow("Cropped image", cropped_image)
        saveimagetofolder(cropped_image,index,path)
    cv2.destroyAllWindows()

# def savecroppedimagesboundaries(image,imgname,array,parent_dir,tw,th):
#     print("saving images")
#     id=findwidthheight(array)
#     w=id[0]-tw
#     h=id[1]-th
#     directory= "ICthresh"+str(imgname)+"/"
#     path = os.path.join(parent_dir, directory)
#     exists = os.path.exists(path)
#     if not exists:
#         os.mkdir(path)

#     for index,item in enumerate(array):

#         i=item[0]
#         j=item[1]
#         cropped_image= image[j:j+h, i:i+w]
#         cv2.imshow("Cropped image", cropped_image)
#         saveimagetofolder(cropped_image,index,path)

#     print("crop image to be displayed")
#     cv2.destroyAllWindows()

