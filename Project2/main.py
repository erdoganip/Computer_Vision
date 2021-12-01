#Name-Surname: İpek Erdoğan
#Computer Vision HW2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math
import scipy
from scipy.ndimage import rotate
import scipy.interpolate

def clicked(image, k):
    plt.imshow(image)
    points = plt.ginput(k, show_clicks=True)
    return [(math.floor(x[0]), math.floor(x[1])) for x in points] #in shape of (x,y) yani (weight,height)

def homography(points1,points2):
    A = []
    for i in range(len(points1)):
        x1, y1 = points1[i][0], points1[i][1]
        x2, y2 = points2[i][0], points2[i][1]
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.asarray(A)
    u, s, vt = np.linalg.svd(A)
    h = vt[-1].reshape(3, 3)
    return h

def opencv_hom(points1, points2):
    h, status = cv2.findHomography(points1, points2)
    return h

def interpolation(image):
    # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
    mask = ~((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))

    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

    # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
    image0 = np.ravel(image[:, :, 0][mask])
    image1 = np.ravel(image[:, :, 1][mask])
    image2 = np.ravel(image[:, :, 2][mask])

    # three separate interpolators for the separate color channels
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, image0)
    interp1 = scipy.interpolate.NearestNDInterpolator(xym, image1)
    interp2 = scipy.interpolate.NearestNDInterpolator(xym, image2)

    # interpolate the whole image, one color channel at a time
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    result1 = interp1(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    result2 = interp2(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

    # combine them into an output image
    result = np.dstack((result0, result1, result2))

    return result

def cropping(image):
    mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coords = np.argwhere(mask != 0)
    p0, pl0 = coords.min(axis=0)
    p1, pl1 = coords.max(axis=0) + 1
    cropped = image[p0:p1, pl0:pl1]
    return cropped

def warp(image,homography):
    emptyImage = np.zeros((2000, 2000, 3), np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            a_ = [[i],[j],[1]]
            res = np.matmul(homography,a_)
            res = res/res[2]
            res = np.around(res).astype(int)
            emptyImage[res[0]+900,res[1]+500] = image[i,j]
    return emptyImage

def resize_with_scale(image,scale):
    scale_percent = scale  # percent of original size
    width_ = int(image.shape[1] * scale_percent / 100)
    height_ = int(image.shape[0] * scale_percent / 100)
    dim = (width_, height_)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename = "paris_b.jpg"
    base_image = imread(filename)

    filename2 = "paris_a.jpg"
    add_image1 = imread(filename2)

    filename3 = "paris_c.jpg"
    add_image2 = imread(filename3)

    with open('points1_a.npy', 'rb') as f1: #paris_a en başta seçtiğim
        points1 = np.load(f1)
    with open('points1_b.npy', 'rb') as f2: #paris_b en başta seçtiğim
        points2 = np.load(f2)

    with open('points2_b.npy', 'rb') as f3: #paris_b points for paris_c
        points3_b = np.load(f3)

    with open('points2_c.npy', 'rb') as f4: #paris_c
        points3_c = np.load(f4)


    hom_mat= homography(points2,points1)
    h_ = np.linalg.inv(hom_mat)
    warped = warp(add_image1,h_)
    print(warped.shape)
    plt.imshow(warped, aspect='auto')
    plt.axis('off')
    plt.savefig('warped_a.png')
    cropped = cropping(warped)
    print(cropped.shape)
    plt.imshow(cropped, aspect='auto')
    plt.axis('off')
    plt.savefig('cropped_a.png')
    result = interpolation(cropped)

    print(result.shape)
    plt.imshow(result, aspect='auto')
    plt.axis('off')
    plt.savefig('interpolated_a.png')

    #resized_base = resize_with_scale(base_image,60)
    #resized_add = resize_with_scale(result, 70)

    hom_mat = homography(points3_b, points3_c)
    h_ = np.linalg.inv(hom_mat)
    warped_c = warp(add_image2, h_)
    print(warped_c.shape)
    plt.imshow(warped_c, aspect='auto')
    plt.axis('off')
    plt.savefig('warped_c.png')
    cropped_c = cropping(warped_c)
    print(cropped_c.shape)
    plt.imshow(cropped_c, aspect='auto')
    plt.axis('off')
    plt.savefig('cropped_c.png')
    result_c = interpolation(cropped_c)
    print(result_c.shape)
    plt.imshow(result_c, aspect='auto')
    plt.axis('off')
    plt.savefig('interpolated_c.png')


    resized_base = base_image
    resized_add = result
    resized_add_2 = result_c

    emp_i = np.zeros((2500, 2000, 3), np.uint8)
    x=875
    y=550
    idx = np.s_[y:y + resized_base.shape[0], x:x + resized_base.shape[1]]
    x2 = resized_base.shape[1]-350
    y2=0
    idx2 = np.s_[y2:y2 + resized_add.shape[0], x2:x2 + resized_add.shape[1]]
    x3 = resized_base.shape[1]-520+ 825
    y3 = y + 27
    idx3 = np.s_[y3:y3 + resized_add_2.shape[0], x3:x3 + resized_add_2.shape[1]]


    resized_add = rotate(resized_add, 25, reshape=False)

    emp_i[idx2] = resized_add
    emp_i[idx3] = resized_add_2
    emp_i[idx] = resized_base

    plt.imshow(emp_i, aspect='auto')
    plt.axis('off')
    plt.savefig('blended.png')

    cr_emp_i = cropping(emp_i)
    plt.imshow(cr_emp_i, aspect='auto')
    plt.axis('off')
    plt.savefig('crop_blended.png')

