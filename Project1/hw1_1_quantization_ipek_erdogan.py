#Name_Surname: Ipek Erdogan
#CV_hw1_Chapter1_Color_Quantization

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math
from scipy.spatial import distance
from PIL import Image



def clicked(image, k):
    plt.imshow(image)
    points = plt.ginput(k, show_clicks=True)
    return [(math.floor(x[0]), math.floor(x[1])) for x in points] #in shape of (x,y) yani (weight,height)

def random_c(image, k):
    h_= np.random.uniform(0,image.shape[0],k)
    w_= np.random.uniform(0,image.shape[1],k)
    n_c = list(zip(h_, w_))
    return [(math.floor(x[1]),math.floor(x[0])) for x in n_c] #turning it into (x,y) shape yani (weight,height)

def euclidian(p1,p2):
    return distance.euclidean(p1, p2) #Distance calculation to use in K-means

def init_center_calc(data, centroids):
    return [(data[x[1]][x[0]][0], data[x[1]][x[0]][1], data[x[1]][x[0]][2]) for x in centroids] #R,G,B

def Kmeans(data,k,ifclicked): #Parameters are: image, k and center determining approach. Will we click on the image or select the color centers randomly?
    shape_y = data.shape[0]
    shape_x = data.shape[1]
    new_data = np.zeros((1200, 1920, 3),dtype = np.uint8)
    if (ifclicked=="clicked"):
        centroids = clicked(data,k)
    elif (ifclicked=="random"):
        centroids = random_c(data,k)
    else:
        print("You should enter valid parameters!")
        return
    centers = init_center_calc(data,centroids) #RGB values of the centroids 
    print(centers)
    count=0
    while (count<5): #5 iterations since 10 was taking too much time. 
        clusters = [[] for _ in range(k)]
        for i in range(shape_y):
            for j in range(shape_x):
                distances=[euclidian((data[i][j][0],data[i][j][1],data[i][j][2]),(center[0],center[1],center[2]))for center in centers]
                class_ind = distances.index(min(distances))
                if (count==4): #At the end of the whole process, assigning our image's elements their new color values.
                    new_data[i][j][0]=int(round(centers[class_ind][0]))
                    new_data[i][j][1]=int(round(centers[class_ind][1]))
                    new_data[i][j][2]=int(round(centers[class_ind][2]))
                clusters[class_ind].append((data[i][j][0],data[i][j][1],data[i][j][2]))
        for i in range(len(clusters)):
            centers[i] = tuple([sum(ele) / len(clusters[i]) for ele in zip(*clusters[i])]) #Calculating the new centers
        print(centers)
        count+=1
    return new_data #returning the transformed image

if __name__ == '__main__':
    #If you want to run the K-means function with "clicked" centers, please use the code below. You need to send K-means function a 3rd parameter: "clicked"
    #filename = "2.jpg"
    #image = imread(filename)
    #new_data = Kmeans(image, 4, "clicked")
    #plt.imsave("2_k_4_clicked.png",new_data)

    #If you want to run the K-means function with "random" centers, please use the codes below. You need to send K-means function a 3rd parameter: "random"
    filename = "3.jpg"
    image = imread(filename)
    new_data = Kmeans(image, 32, "random")
    plt.imsave("3_k_32.png",new_data)

    filename2 = "1.jpg"
    image2 = imread(filename2)
    new_data2 = Kmeans(image2, 32, "random")
    plt.imsave("1_k_32.png", new_data2)

