import cv2 as cv
import torch
import numpy as np
from icecream import ic
from PIL import Image
import os
from tqdm.notebook import tqdm



def processed_to_final(image_addr_list, path,min_area, max_area):
    """
    Convert processed denoised images to final defects-only images
    Args:
        image_addr_list (str): 
        path (_type_): _description_
        min_area (_type_): _description_
        max_area (_type_): _description_
    """
    num_defects = []
    # min_area, max_area = 20, 20000
    for i in tqdm(image_addr_list):
        full_path = os.path.join(path, i)
        imgobj = Image.open(full_path).convert('RGB')
        img = np.asarray(imgobj)
        # Convert to grayscale images
        grayimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # Find contours
        contours, hierarchy = cv.findContours(grayimg,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        num_poles = 0
        valid_contours = []
        # Loop through each contour to inspect
        for j in range(len(contours)):
            x, y, w, h = cv.boundingRect(contours[j])
            # Middle point must be black
            if grayimg[int(y+h/2), int(x+w/2)] == 0:
                # Pole can not be too close to the boundary
                if grayimg[int(max(y-h/2, 0)), int(x+w/2)] != 0:
                    # Pole should be large enough but not too big
                    if h*w > min_area and h*w < max_area:
                        num_poles += 1
                        valid_contours.append(contours[j])
        num_defects.append(num_poles)
        # Save to processed
        mask = np.ones(img.shape[:2],dtype=np.uint8) * 255
        img_arr = cv.drawContours(mask, valid_contours, -1, (0,255,255), -1)
        Image.fromarray(img_arr).save('../Data/final/' + i)
# TODO: wrap up code in image_processing.ipynb into this file
if __name__ == '__main__':
    ic("contour.py")
