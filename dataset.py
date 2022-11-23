import torch
import torch.nn as nn
import numpy as np


# Return the feature given any start and end location (0 - 35mm)
# 35mm for spectrum & image
def feature_spectra_window(spec_feat, start_len, end_len, len_spectra=942):
    start_spectra = int(start_len / 35 * len_spectra)
    end_spectra = int(end_len / 35 * len_spectra)
    return(spec_feat[start_spectra: end_spectra, :, :])


def feature_image_window(img_feat, start_len, end_len, len_image=4046):
    start_spectra = int(start_len / 35 * len_image)
    end_spectra = int(end_len / 35 * len_image)
    return(img_feat[start_spectra: end_spectra, :])


def get_img_spec_window(spec_feat, img_feat, start_len, end_len):
    print(f"start: {np.round(start_len, 2)}mm | end: {np.round(end_len, 2)}mm")
    spec = feature_spectra_window(spec_feat, start_len, end_len)
    print(f" -- spectra window feature shape: {spec.shape}")
    print(f" -- spectra window size: {spec.shape[0]}")
    img = feature_image_window(img_feat, start_len, end_len)
    print(f" -- image window feature shape: {img.shape}")
    print(f" -- image window size: {img.shape[0]}")
    return spec, img


def preprocess_img_spec_tuple(spec_feat, img_feat, start_len, end_len, step_size):
    cur_start = start_len
    cur_end = cur_start + step_size
    while cur_end <= end_len:
        spec, img = get_img_spec_window(
            spec_feat, img_feat, cur_start, cur_end)
        cur_start += step_size
        cur_end += step_size
