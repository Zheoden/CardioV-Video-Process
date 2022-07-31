from ast import main
import cv2 as cv
import os
import numpy as np
import sys

def resize_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def get_img_borders(img):
    """Function that returns a processed image, showing only its edges

    Args:
        path (_type_): image path
    """
    img_resized = resize_frame(img)
    # Blurring may be necessary to not have extra edges
    blur = cv.GaussianBlur(img_resized, (3,3), cv.BORDER_DEFAULT)
    # Canny edge detecting:
    edges = cv.Canny(blur, 125, 175)
    
    return edges
    
def process_video(path):
    """Function that returns the list of the video frames. They will br processed, showing only their edges

    Args:
        path (_type_): Path to the video
    """
    vid = cv.VideoCapture(path)
    video_frames = []
    success,image = vid.read()
    while success:
        success, frame = vid.read()
        frame_edges = get_img_borders(frame)
        video_frames.append(frame_edges)
    vid.release()
    
    return video_frames

def process_image(path):
    img = cv.imread(path)
    img_resized = resize_frame(img)
    img_edges = get_img_borders(img_resized)

def show_frames(list):
    counter = 1
    for frame in list:
        cv.imshow(f'Heart_{counter}',frame)
        cv.waitKey(0)
        counter += 1

def show_img(img):
    cv.imshow('Heart',img)
    cv.waitKey(0)

if __name__ == "__main__":
    
    if sys.argv[1] == 'v':
        frame_list = process_video(sys.argv[2])
        show_frames(frame_list)
    elif sys.argv[1] == 'i':
        img = process_image(sys.argv[2])
        show_img(img)
    