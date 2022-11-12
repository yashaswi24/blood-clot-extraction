from PIL import Image

import cv2

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import imag

path="/Users/yashaswiveerepalli/Documents/Fall22/Data Mining/yashaswi_veerepalli_assignment1/slices/ICthresh7/12slice.jpg"

import colorgram

# It is the number of colors you want to extract from the image
nb_colors = 30      

# Returns a list of tuples form (r, g, b) 
image_colors = colorgram.extract(path, nb_colors)   

print(len(image_colors))
image = plt.imread(path)

# red, blue, green = image[:,:,0], image[:,:,1], image[:,:,2]
# print(red)
# print(green)
# print(blue)


thresholded=cv2.inRange(image,(0,0,0),(0,0,0))
#add both images
res=image+cv2.cvtColor(thresholded,cv2.COLOR_GRAY2BGR)

cv2.imshow("img",image)
cv2.waitKey(10000)




# img = Image.open(path)
# img = img.convert("RGB")

# d = img.getdata()

# new_image = []
# for item in d:

# 	# change all white (also shades of whites)
# 	# pixels to yellow
# 	if item[0] in list(range(200, 256)):
# 		new_image.append((255, 224, 100))
# 	else:
# 		new_image.append(item)
		
# # update image data
# img.putdata(new_image)

# # save new image
# cv2.imshow("img",img)
# cv2.waitKey(10000)
