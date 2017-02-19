import cv2, os
import numpy as np
from skimage.exposure import adjust_gamma
#from scipy import ndimage
#import pandas as pd

# crop images, default is NVIDIA model image/imputs
def crop_image(image, crop_height = 66, crop_width = 200, x_start = 60, y_start = 70):
    height = image.shape[0]
    width = image.shape[1]
    return image[y_start : y_start + crop_height, x_start : x_start + crop_width]

def flip_image(image):
    return cv2.flip(image, 1)

def resize_image(image, image_height=66, image_width=200):
    return cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def adjust_brightness(image, ratio):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def translate_image(image, x_trans, y_trans):
    m_trans = np.float32([[1, 0, x_trans], [0, 1, x_trans]])
    height, width = image.shape[:2]
    return cv2.warpAffine(image, m_trans, (width, height))

def preprocess(image):
    #image = crop_image(image)
    #image = resize_image(image)
    #image = rgb2yuv(image)
    #image = adjust_gamma(image)
    return image

def random_flip(image, steering):
    if np.random.rand() < 0.5:
        image = flip_image(image)
        steering = -steering
    return image, steering

def random_brightness(image):
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    return adjust_brightness(image, ratio)

def random_translate(image, steering, x_range, y_range):
    x_trans = x_range * (np.random.rand() - 0.5)
    y_trans = y_range * (np.random.rand() - 0.5)
    steering = steering + x_trans * 0.002
    image = translate_image(image, x_trans, y_trans)
    return image, steering

def random_shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)] = 1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1] * random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0] * random_bright        
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def random_effects(image, steering):
    image, steering = random_flip(image, steering)
    image, steering = random_translate(image, steering, 100, 10)
    image = random_brightness(image)
    image = random_shadow(image)
    return image, steering

def get_image_file_name(path):
    path = os.path.normpath(path)
    path_split = path.split(os.sep)
    return path_split[-1]

def load_image(image_file_name):
    img = cv2.imread(image_file_name.strip())
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #return ndimage.imread(image_file_name.strip()).astype(np.float32)

def get_proceesed_data(image_file_name, steering, data_augmentation):
    image = load_image(image_file_name)
    # Preprocess image
    image = preprocess(image)
    # Add random effects for train data only
    if data_augmentation == True:
        if np.random.rand() < 0.5:
            image, steering = random_effects(image, steering)
    return image, steering