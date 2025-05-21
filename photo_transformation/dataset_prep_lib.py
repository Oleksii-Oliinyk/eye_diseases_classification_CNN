import cv2
import os
import random 
import numpy as np

def read_images(read_path):    
    images = []
    for filename in os.listdir(read_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(read_path, filename))
        if img is not None:
            images.append(img)
    print(f"Loaded {len(images)} images from {read_path}")
    return images

def process_validation_images(images): 
    i = 0
    result_images = []
    for image in images: 
        i+=1
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray_img.shape
        if height == width:
            cropped_img = gray_img
        a = abs(height - width)
        a1 = a // 2
        a2 = a1 if a % 2 == 0 else a1 + 1
        if height > width:
            cropped_img = gray_img[a1:height-a2, 0:width]
        if height < width:
            cropped_img = gray_img[0:height, a1:width-a2]
        resized_img = cv2.resize(cropped_img, (128, 128))
        result_images.append(resized_img)

    print("All photos were processed!!!")
    return result_images

def process_retinoblastoma_images(images): 
    processed_images = []
    i = 0
    for image in images: 
        i+=1
        height, width, _ = image.shape
        if height == width:
            cropped_img = image
        a = abs(height - width)
        a1 = a // 2
        a2 = a1 if a % 2 == 0 else a1 + 1
        if height > width:
            cropped_img = image[a1:height-a2, 0:width]
        if height < width:
            cropped_img = image[0:height, a1:width-a2]
        
        processed_img = cv2.resize(cropped_img, (256, 256), interpolation = cv2.INTER_AREA)
        
        processed_images.append(processed_img)

    return processed_images

def process_classification_images(images): 
    processed_images = []
    i = 0
    for image in images: 
        i+=1
        height, width, _ = image.shape
        if height == width:
            cropped_img = image
        a = abs(height - width)
        a1 = a // 2
        a2 = a1 if a % 2 == 0 else a1 + 1
        if height > width:
            cropped_img = image[a1:height-a2, 0:width]
        if height < width:
            cropped_img = image[0:height, a1:width-a2]
        
        processed_img = cv2.resize(cropped_img, (640, 640), interpolation = cv2.INTER_AREA)
        
        processed_images.append(processed_img)

    return processed_images


def flip_images(images):
    flipped_images = []
    for image in images:
        flipped_img = cv2.flip(image, flipCode=1)
        flipped_images.append(flipped_img)
    return flipped_images
    
def rotate_images(images):
    rotated_images = []
    for image in images:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        mean_color = image.mean(axis=(0, 1)).astype(np.uint8)

        angle = round(random.uniform(-21, 21), 1)  
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_img = cv2.warpAffine(image, M, (w, h), borderValue=mean_color.tolist())
        rotated_images.append(rotated_img)
    
    return rotated_images

def save_images(images, save_path, name):
    i = 0
    for img in images:
        i += 1
        filename = f"{name}_{i}.png"
        cv2.imwrite(os.path.join(save_path, filename), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"Photo {filename} saved to {save_path}")
    print("All photos were saved!!!")
