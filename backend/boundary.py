import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
from image_registration import chi2_shift
from scipy.ndimage import shift


def resize_image(image):
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC)

def rotate_image(image):
    return cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

def color_it_black(image):
    np_image = np.array(image)
    for i in range(np_image.shape[0]):
        for j in range(np_image.shape[1]):
            # np_image[i,j] = 0 if np_image[i,j] < 200 else 255
            np_image[i,j] = 255 if np_image[i,j] < 200 else 0
    return np_image

def masking(image):
    np_image = np.array(image)
    for i in range(np_image.shape[0]):
        for j in range(np_image.shape[1]):
            np_image[i,j] = 0 if np_image[i,j] == 255 else 1
    return np_image

def white_to_black(image):
    np_image = np.array(image)
    for i in range(np_image.shape[0]):
        for j in range(np_image.shape[1]):
            np_image[i,j] = 255 if np_image[i,j] == 0 else 0
    return np_image


def fit_boundary(generated_path, boundary):
    final_image = None
    max_score = 80
    generated_image = cv2.imread(generated_path)
    gen_grayimage = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
    w, h = gen_grayimage.shape
    offset_image = cv2.imread(boundary)
    offset_image = cv2.cvtColor(offset_image, cv2.COLOR_BGR2GRAY)
    offset_image = cv2.resize(offset_image, (h,w))
    # Resize both images to same shape
    offset_image = resize_image(offset_image)
    gen_grayimage = resize_image(gen_grayimage)
    for item in range(4):
        
        image = rotate_image(gen_grayimage)
        # conver to complete black
        # offset_image = color_it_black(offset_image)
        image = color_it_black(image)
        # print(image.shape, offset_image.shape)

        #Method 1: chi squared shift
        #Find the offsets between image 1 and image 2 using the DFT upsampling method
        noise=0.1
        xoff, yoff, exoff, eyoff = chi2_shift(image, offset_image, noise, 
                                            return_error=True, upsample_factor='auto')

        # print("Pixels shifted by: ", xoff, yoff)

        corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

        original = white_to_black(image) #cv2.imread(file1)
        duplicate = white_to_black(corrected_image)
        difference = cv2.subtract(original, duplicate)
        # Inverse difference
        difference_2 = cv2.subtract(original, duplicate)
        # Maching Score
        score = ((1 - difference.sum()/original.sum()) + (1 - difference_2.sum()/duplicate.sum()))*100/2
        if score >= max_score:
            max_score = score
            final_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
            # final_image = resize_image(final_image)
            # duplicate = masking(duplicate)
            # for i in range(3):
            #     final_image[:,:,i] = duplicate * final_image[:,:,i]
            final_image = Image.fromarray(final_image)
    if final_image != None:
        final_image.save(generated_path, "jpeg")  
    if max_score > 80:
        return max_score

    return 0


 

