import cv2, os
import numpy as np
from skimage.exposure import adjust_gamma

# crop images, default is NVIDIA model image/imputs
def crop_image(image, crop_height = 66, crop_width = 200, x_start = 60, y_start = 70):
    height = image.shape[0]
    width = image.shape[1]
    return image[y_start : y_start + crop_height, x_start : x_start + crop_width]

# flip image, horizontal
def flip_image(image):
    return cv2.flip(image, 1)

# resize image
def resize_image(image, image_height=66, image_width=200):
    return cv2.resize(image, (image_width, image_height), cv2.INTER_AREA)

# RGB to YUV
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# brighness adjustment
def adjust_brightness(image, ratio):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# translate image
def translate_image(image, x_trans, y_trans):
    m_trans = np.float32([[1, 0, x_trans], [0, 1, x_trans]])
    height, width = image.shape[:2]
    return cv2.warpAffine(image, m_trans, (width, height))

# preprocess
def preprocess(image):
    #image = crop_image(image)
    #image = resize_image(image)
    #image = rgb2yuv(image)
    #image = adjust_gamma(image)
    return image

# random image flip (horizontal)
def random_flip(image, steering):
    if np.random.rand() < 0.5:
        image = flip_image(image)
        steering = -steering
    return image, steering

# random brightness adjustment
def random_brightness(image):
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    return adjust_brightness(image, ratio)

# random gamma correction
def random_gamma(image):
    correction = 1.0 + 0.4 * (np.random.rand() - 0.5)
    return adjust_gamma(image, gamma=correction)

# random translation effect
def random_translate(image, steering, x_range, y_range):
    x_trans = x_range * (np.random.rand() - 0.5)
    y_trans = y_range * (np.random.rand() - 0.5)
    steering = steering + x_trans * 0.002
    image = translate_image(image, x_trans, y_trans)
    return image, steering

# random shadow effect
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

# random addition of image effects
def random_effects(image, steering):
    image, steering = random_flip(image, steering)
    image, steering = random_translate(image, steering, 100, 10)
    image = random_brightness(image)
    image = random_shadow(image)
    image = random_gamma(image)
    return image, steering

# get the image file name from a path
def get_image_file_name(path):
    path = os.path.normpath(path)
    path_split = path.split(os.sep)
    return path_split[-1]

# load image 
def load_image(image_file_name):
    img = cv2.imread(image_file_name.strip())
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# get processed image, adjusted steering value for a given image file, steering inputs
def get_proceesed_data(image_file_name, steering, data_augmentation):
    image = load_image(image_file_name)
    # Preprocess image
    image = preprocess(image)
    # Add random effects for train data only
    if data_augmentation == True:
        if np.random.rand() < 0.5:
            image, steering = random_effects(image, steering)
    return image, steering