import math
import cv2
import numpy as np

def get_combined_img(img, grasp_img):    
    img_shape =  img.shape
    grasp_shape = grasp_img.shape

    img = np.nan_to_num(img, nan=0)
    grasp_img = np.nan_to_num(grasp_img, nan=0)

    if len(img_shape) != 3:
        img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_BONE)

    if len(grasp_shape) != 3:
        grasp_img = cv2.applyColorMap((grasp_img * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    combined_img = np.zeros((img_shape[0], img_shape[1] + grasp_shape[1] + 10, 3), np.uint8)
    combined_img[:img_shape[0], :img_shape[1]] = img
    combined_img[:grasp_img.shape[0], img_shape[1]+10:img_shape[1]+grasp_shape[1]+10] = grasp_img

    return combined_img
