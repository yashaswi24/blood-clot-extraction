import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import cv2
import extcolors
from colormap import rgb2hex

path='/Users/yashaswiveerepalli/Documents/Fall22/Data Mining/yashaswi_veerepalli_assignment1/slices/ICthresh7/12slice.jpg'
# Reading an image
# img = Image.open(path,'r')

# colors_x = extcolors.extract_from_path(img, tolerance = 12, limit = 12)
# print(colors_x)

image_mark= cv2.imread(path)
image= cv2.imread(path)

lower_white= np.array([0,0,255])
upper_white= np.array([255,255,255])
mask = cv2.inRange(image_mark, lower_white, upper_white)
mask_inv = cv2.bitwise_not(mask)
 
    # Convert to grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # Extract the dimensions of the original image
rows, cols, channels = image.shape
image = image[0:rows, 0:cols]
 
    # Bitwise-OR mask and original image
colored_portion = cv2.bitwise_or(image, image, mask = mask)
colored_portion = colored_portion[0:rows, 0:cols]
 
    # Bitwise-OR inverse mask and grayscale image
gray_portion = cv2.bitwise_or(gray, gray, mask = mask_inv)
gray_portion = np.stack((gray_portion,)*3, axis=-1)
 
    # Combine the two images
output = colored_portion + gray_portion

cv2.imshow("Image ", colored_portion)
cv2.waitKey(0)
    