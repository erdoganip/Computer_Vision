#Name_Surname: Ipek Erdogan
#CV_hw1_Chapter2_Connected_Component_Analysis

import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import numpy as np

# Birds1: Class: 18 Blur+Closing with 4,4 kernel have been used.
# Birds2: Class: 14 Blur+Opening with 10,9 kernel have been used.
# Birds3: Class: 14 Blur+Erosion with 9,9 kernel have been used.
# Dice5: Class: 6 Blur + Dilation + Erosion + Opening with kernel 10,10 have been used. Threshold was zero to get rid of the shadowed background.
# Dice6: Class: 6 Dilation + Erosion + Opening with kernel (9,10) have been used. Threshold was zero to get rid of the shadowed background.
# Dice6-2: Class: 6 Blur + Dilation + Erosion + Opening with kernel (9,10) have been used. Threshold was zero to get rid of the shadowed background.


if __name__ == '__main__':

    img = cv2.imread('dice6_2.png', 0)
    ret, thresh = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV)  # 127

    cv2.imwrite('binary.png', thresh * 255) #Saving the "binary" version of the image

    #I used different morphological operations. These operation functions are down below:
    
    kernel = np.ones((9, 10), np.uint8)
    thresh2 = cv2.blur(thresh,(5,5))
    # thresh2 = cv2.medianBlur(thresh,5)

    thresh3 = cv2.dilate(thresh2, kernel, iterations=1)  # dilation
    thresh4 = cv2.erode(thresh3, kernel, iterations=1)  # erosion
    thresh1 = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel)  # opening: erosion followed by dilation.
    # thresh1 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel) #closing: dilation followed by erosion
    # thresh1 = cv2.morphologyEx(thresh3, cv2.MORPH_TOPHAT, kernel) #tophat
    # thresh1 = cv2.morphologyEx(thresh3, cv2.MORPH_BLACKHAT, kernel) #blackhat

    cv2.imwrite('binary_processed.png', thresh1 * 255) #This is the input of the connected component analysis function

    labels = {} #Keeping labels and positions in a dictionary. Positions=keys, Labels=Values
    for i in range(thresh1.shape[0]):
        for j in range(thresh1.shape[1]):
            if (thresh1[i][j] == 0): #Labeling background pixels with "0"
                labels[(i, j)] = 0
            else:
                labels[(i, j)] = -1 #Labeling foreground pixels with "-1"

#CONNECTED COMPONENT ANALYSIS
    new_label_count = 5 #Starting labeling with "5"

    for i in range(thresh1.shape[0]):
        for j in range(thresh1.shape[1]):
            neighbor_labels = {} #Creating a neighbor labels dictionary for all of the elements in the image
            if (labels[(i, j)] == -1):
                if (i != 0):
                    neighbor_labels[(i - 1, j)] = labels[(i - 1, j)]
                if (i != 0 and j != thresh1.shape[1] - 1):
                    neighbor_labels[(i - 1, j + 1)] = labels[(i - 1, j + 1)]
                if (i != 0 and j != 0):
                    neighbor_labels[(i - 1, j - 1)] = labels[(i - 1, j - 1)]
                if (i != thresh1.shape[0] - 1):
                    neighbor_labels[(i + 1, j)] = labels[(i + 1, j)]
                if (i != thresh1.shape[0] - 1 and j != 0):
                    neighbor_labels[(i + 1, j - 1)] = labels[(i + 1, j - 1)]
                if (j != 0):
                    neighbor_labels[(i, j - 1)] = labels[(i, j - 1)]
                if (j != thresh1.shape[1] - 1):
                    neighbor_labels[(i, j + 1)] = labels[(i, j + 1)]
                if (i != thresh1.shape[0] - 1 and j != thresh1.shape[1] - 1):
                    neighbor_labels[(i + 1, j + 1)] = labels[(i + 1, j + 1)]

                #To see if there is an already labeled neighbor
                candidates = {k: v for k, v in neighbor_labels.items() if (v != 0 and v != -1)} 

                #If there is no already labeled neighbor around, we create a new class label with our counter "new_label_count"
                if (len(candidates) == 0):
                    labels[(i, j)] = new_label_count
                    for k, v in neighbor_labels.items():
                        if v == -1:
                            labels[(k[0], k[1])] = new_label_count
                    new_label_count += 1

                #If there is just one labeled neighbor around, we assign this label to our element
                elif (len(candidates) == 1):
                    for k, v in candidates.items():
                        labels[(i, j)] = v

                #If there are different labels in our element's neighbors, it means we need to equialize these labels.
                #We are taking the minimum valued label as our base and equalize our "other labeled" neighbors to this minimum label
                else:
                    min_ = min(candidates.items(), key=lambda x: x[1])[1]
                    labels[(i, j)] = min_
                    for k, v in neighbor_labels.items():
                        if (v == -1 or v > min_):
                            labels[(k[0], k[1])] = new_label_count

    #Here you can print the different classes
    # for k,v in labels.items():
    # if(v!=0 and v!=-1):
    # print(v)

    #Taking the maximum value of the class count
    max_ = max(labels.items(), key=lambda x: x[1])[1]
    #And subtracting 5 and adding 1 to find the class (object) count
    print(max_ - 4) #prints the object count
