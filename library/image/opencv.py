# Python Imports 
import os
import sys
import json
import math
import random
import warnings

# Library Imports
import cv2
import numpy as np

# Custom Imports

# Gloabal Variable/Settings
warnings.filterwarnings("ignore")

def resize_image_opencv_pad(input_image, output_image, resize_image_width, resize_image_height, color=(255,255,255)):

    img = cv2.imread(input_image)
    
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color for padding
    result = np.full((resize_image_height, resize_image_width, channels), color, dtype=np.uint8)

    # copy img image into top right corner of result (plain white image) image
    result[0:old_image_height, 0:old_image_width] = img

    cv2.imwrite(output_image, result)

def resize_image_opencv(input_image, output_image, resize_to_width, resize_to_height):

    img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[:2]
    pad_bottom, pad_right = 0, 0
    ratio = w / h

    if h > resize_to_height or w > resize_to_width:
        # shrinking image algorithm
        interp = cv2.INTER_AREA
    else:
        # stretching image algorithm
        interp = cv2.INTER_CUBIC

    w = resize_to_width
    h = round(w / ratio)
    if h > resize_to_height:
        h = resize_to_height
        w = round(h * ratio)
    pad_bottom = abs(resize_to_height - h)
    pad_right = abs(resize_to_width - w)

    scaled_img = cv2.resize(img, (w, h), interpolation=interp)
    padded_img = cv2.copyMakeBorder(scaled_img,0,pad_bottom,0,pad_right,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])

    cv2.imwrite(output_image, padded_img)

def resize_all_opencv(input_path, output_path, resize_to_width, resize_to_height, allowed_image_extensions):

    # get all the pictures in directory
    images = []

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(tuple(allowed_image_extensions)):
                images.append(os.path.join(root, file))

    for image in images:

        filename_ext = os.path.basename(image).split(".")

        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        h, w = img.shape[:2]
        pad_bottom, pad_right = 0, 0
        ratio = w / h

        if h > resize_to_height or w > resize_to_width:
            # shrinking image algorithm
            interp = cv2.INTER_AREA
        else:
            # stretching image algorithm
            interp = cv2.INTER_CUBIC

        w = resize_to_width
        h = round(w / ratio)
        if h > resize_to_height:
            h = resize_to_height
            w = round(h * ratio)
        pad_bottom = abs(resize_to_height - h)
        pad_right = abs(resize_to_width - w)

        scaled_img = cv2.resize(img, (w, h), interpolation=interp)
        padded_img = cv2.copyMakeBorder(
            scaled_img,0,pad_bottom,0,pad_right,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
        
        cv2.imwrite(os.path.join(output_path, filename_ext[0]+"_resized."+filename_ext[1]), padded_img)

def display_images_from_path(config, path, log):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(tuple(config["files"]["allowed_image_extensions"])):
                img = cv2.imread(os.path.join(root,file))
                cv2.imshow(config["opencv"]["window_name"], img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()