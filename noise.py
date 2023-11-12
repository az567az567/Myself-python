import os
import cv2
import numpy as np
import random
import math

def sp_noise(noise_img, proportion):
    height, width = noise_img.shape[0],noise_img.shape[1]
    num = int(height * width * proportion)
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0,height - 1)
        if random.randint(0,1) == 0:
            noise_img[h,w] = 0
        else:
            noise_img[h,w] = 255
    return noise_img

def gaussian_noise(img,mean,sigma):
    img = img / 255
    noise = np.random.normal(mean,sigma,img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    return gaussian_out

def random_noise(image,noise_num):
    img_noise = image
    rows, cols, chn = img_noise.shape
    for i in range(noise_num):
        x = np.random.randint(0,rows)
        y = np.random.randint(0, cols)
        img_noise[x, y, : ] = 255
    return img_noise

def modify_contrast_and_brightness2(img, brightness , contrast):
    img_noise = img
    B = brightness/255.0
    c = contrast/255.0
    k = math.tan((45+44*c)/180*math.pi)
    img_noise = (img - 127.5 * (1-B))* k +127.5 * (1+B)
    # img = np.clip(img, 0, 255).astype(np.uint8)
    return img_noise

def flip(image):
    img_noise = image
    img_noise = cv2.flip(image, 1)
    return img_noise

def rotate_img(img):
    (h, w, d) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -50, 1.0)
    img_noise = cv2.warpAffine(img, M, (w, h))
    return img_noise

def convert(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        path = input_dir + "/" + filename
        print("doing...",path)
        noise_img = cv2.imread(path)
        #img_noise = gaussian_noise(noise_img,0,0.12) #11
        #img_noise = sp_noise(noise_img,0.025) #12
        #img_noise = random_noise(noise_img,1500) #14 x @@@
        #img_noise = modify_contrast_and_brightness2(noise_img, 0 ,-100) #13 #40+100
        #img_noise = modify_contrast_and_brightness2(noise_img, 0 ,-90) #16 #40+100
        #img_noise = modify_contrast_and_brightness2(noise_img, 0, +50) #17
        #img_noise = modify_contrast_and_brightness2(noise_img, 0, +100)  # 18
        img_noise = flip(noise_img) #15 @@@
        #img_noise = rotate_img(noise_img)#36
        cv2.imwrite(output_dir + '/'+ filename,img_noise )

if __name__ == '__main__':
    input_dir = "C:\\Users\\az567\\Desktop\\dataset(3)\\yolov7datasetdataset(3)2\\valid\\JPEGImages"
    output_dir = "C:\\Users\\az567\\Desktop\\dataset(3)\\yolov7datasetdataset(3)2\\valid\\2"
    convert(input_dir,output_dir)
