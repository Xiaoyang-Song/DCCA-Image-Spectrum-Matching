from multiprocessing import ProcessError
import cv2 as cv
import torch
import numpy as np
from icecream import ic
from PIL import Image
import os
from tqdm.notebook import tqdm

# A Wrapper for functions illustrated in image_processing.ipynb


def raw_to_processed(raw_path, processed_path,
                     clip_dict={'hmin': 0, 'hmax': 250, 'wmin': 120, 'wmax': 850}):
    """
    Convert raw data into clipped & denoised images.
    Args:
        raw_path (str): path to raw images (locally "../Data/raw")
        processed_path (str): saving path to processed images (locally "../Data/processed")
        clip_dict (dict, optional): _description_. 
            Defaults to {'hmin':0, 'hmax': 250, 'wmin':120, 'wmax':850}.
    """
    image_addr_list = os.listdir(raw_path)
    hmin = clip_dict['hmin']
    hmax = clip_dict['hmax']
    wmin = clip_dict['wmin']
    wmax = clip_dict['wmax']
    print("Processing raw images by clipping and denoising ...")
    for i in tqdm(image_addr_list):
        full_path = os.path.join(raw_path, i)
        img = Image.open(full_path).convert('RGB')
        img_array = np.asarray(img)
        # Clip
        img_ = Image.fromarray(img_array[hmin:hmax, wmin:wmax], 'RGB')
        img_ = cv.cvtColor(np.asarray(img_), cv.COLOR_BGR2GRAY)
        # Denoise
        _, thresh1 = cv.threshold(
            img_, 255, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Save
        Image.fromarray(thresh1).save(processed_path + i)


def processed_to_final(processed_path, save_path, pole_dict={'min_a': 20, 'max_a': 20000}):
    """
    Convert processed denoised images to final defects-only images
    Args:
        processed_path (str): path to processed data (locally "../Data/processed/")
        save_path (str): path for saving final data (locally "../Data/final/")
        min_area (float): minimum area for poles
        max_area (float): maximum area for poles
    """
    num_defects = []
    image_addr_list = os.listdir(processed_path)
    min_area, max_area = pole_dict['min_a'], pole_dict['max_a']
    print("Extract poles-only (i.e. defects-only) images by heuristic contour finding ...")
    for i in tqdm(image_addr_list):
        full_path = os.path.join(processed_path, i)
        imgobj = Image.open(full_path).convert('RGB')
        img = np.asarray(imgobj)
        # Convert to grayscale images
        grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find contours
        contours, hierarchy = cv.findContours(
            grayimg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        img_arr = cv.drawContours(mask, valid_contours, -1, (0, 255, 255), -1)
        Image.fromarray(img_arr).save(save_path + i)


def one_step_process(raw_path, processed_path, final_path,
                     pole_dict={'min_a': 20, 'max_a': 20000},
                     clip_dict={'hmin': 0, 'hmax': 250, 'wmin': 120, 'wmax': 850}):
    raw_to_processed(raw_path, processed_path, clip_dict)
    processed_to_final(processed_path, final_path, pole_dict)


if __name__ == '__main__':
    ic("process.py")
