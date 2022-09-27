import cv2 as cv
#import os
import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import imutils
import math
import urllib
#import wall_thikness_calculator as wtc

_CENTIMETERS = 0.2

def process_video(path, show_images = False):
    """Function that returns the list of the video frames. They will br processed, showing only their edges

    Args:
        path (_type_): Path to the video
    """
    vid = cv.VideoCapture(path)
    video_frames = []
    success, frame = vid.read()
    
    area_mask = get_area_of_interest(frame, show_images)
    while success:        
        processed_frame = process_image(frame, area_mask, show_images)
        video_frames.append(processed_frame)
        success, frame = vid.read()
        if not success:
            print('End of video')
            break
        
    vid.release()

    return video_frames

def process_image(img, area_mask, show_images = False):
    """
        Process image first read the img from the path provided
        Then it follows these next steps:
        1- Apply color filter to get clean areas and a black and white img
        3- Call function get borders
        4- Call function get contours 
    """    
    show_img(img, 'Heart Original') if show_images else 1
    
    x, y, w, h = cv.boundingRect(area_mask)
    gray = get_grays(img)
    result = cv.bitwise_and(gray, gray, mask = area_mask)
    cropped = result[y:y+h,x:x+w]
    
    
    show_img(cropped,"Areas in black and white") if show_images else 1
    
    img_edges = get_img_borders(cropped)
    show_img(img_edges,'Heart Edges') if show_images else 1

    img_processed, contours = get_img_contour(img_edges)
    show_img(img_processed,'Heart contours') if show_images else 1

    return img_edges
    
def process_selector(type, path, show_images= False):
    
    ret_val = 'hi'
    if type == 'v':
        print('initializing video processing')
        frame_list = process_video(path, show_images)
        ret_val = 'Video processed'
        
    elif type == 'i':
        print('initializing img processing. Path:')
        print(path)
        resp = urllib.request.urlopen(path)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img_to_process = cv.imdecode(image, cv.IMREAD_COLOR)
        #img_to_process = cv.imread(path)
        mask = get_area_of_interest(img_to_process, show_images)
        img = process_image(img_to_process, mask, show_images)
        ret_val = 'Image processed'
        
    return ret_val

def process_values(path, type= 'i'):
    
    if(type == 'i'):
        print('initializing img volume estimation. URL:')
        print(path)
        resp = urllib.request.urlopen(path)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img_to_process = cv.imdecode(image, cv.IMREAD_COLOR)
        data_set = estimate_img_values(img_to_process)
        
    elif(type == 'v'):
        print(path)
        data_set = estimate_video_values(path)
    
    return data_set
