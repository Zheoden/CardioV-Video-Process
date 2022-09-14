import cv2 as cv
#import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import imutils
from stack_images import stack_images

def resize_frame(frame, scale=0.75):
    
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

def get_img_borders(img):
    """Function that returns a processed image, showing only its edges

    Args:
        path (_type_): image path
    """
    
    img_resized = resize_frame(img)
    # Blurring may be necessary to not have extra edges
    blur = cv.GaussianBlur(img_resized, (3,3), cv.BORDER_DEFAULT)
    show_img(blur)
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
        if not success:
            print('End of video')
            break
        frame_edges = get_img_borders(frame)
        video_frames.append(frame_edges)
    vid.release()
    
    return video_frames

def process_image(path):
    
    img = cv.imread(path)
    print('showing original img')
    cv.imshow('Heart',img)
    cv.waitKey(0)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("applying thrershold")
    thresh, thresh_img = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
    rgb_img = cv.cvtColor(thresh_img, cv.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15,15))
    plt.imshow(rgb_img)
    plt.show()
    img_resized = resize_frame(gray)
    img_edges = get_img_borders(img_resized)
    
    
    print('Img processed!')
    
   # return img_edges
    return img_edges

def get_img_contour(img):
    
    print('finding countours')
    contours, hierarchies = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    print('creating blank img')
    blank = np.zeros(img.shape, dtype = 'uint8')
    print('writing contours')
    print(f'{len(contours)} contours found')
    # -1 is to use every contour detected
    cv.drawContours(blank, contours, -1, (255,0,255), 2)
    conts = imutils.grab_contours((contours,hierarchies))
    print(f"imutils conts = {len(conts)}")
    return blank,contours
    
def calculate_perimeter(cont):
   
    for item in cont:
        perimeter = cv.arcLength(item,True)
        print(f"PERIMETER = {perimeter}")
    

def show_frames(list):
    
    counter = 1
    total = len(list)
    for frame in list:
        print(f'Showing img {counter} of {total}')
        cv.imshow(f'Heart_{counter}',frame)
        cv.waitKey(0)
        counter += 1

def show_img(img):
    
    print('Showing image')
    cv.imshow('Heart',img)
    cv.waitKey(0)
    
    
def measure_contours(img, contours):
    
    for cont in contours:
        # extracting box points
        print("")
        
def get_blue(path):
    
    img = cv.imread(path)
    
    
    # It converts the BGR color space of image to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_gray = np.array([50,100,50])
    upper_gray = np.array([70,255,255])

    mask2 = cv.inRange(hsv, lower_gray, upper_gray)
    resulthsv = cv.bitwise_and(img, img, mask = mask2)
    
    
    blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
    # Threshold of blue in HSV space
    lower_blue = np.array([241, 35, 140])
    upper_blue = np.array([255, 255, 255])
    # preparing the mask to overlay
    mask = cv.inRange(blur, lower_blue, upper_blue)
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv.bitwise_and(img, img, mask = mask)

    img_stack = stack_images(0.8,([blur, mask, result]))
    img_stack2 = stack_images(0.8,([hsv, mask2, resulthsv]))

    cv.imshow('img_stack', img_stack)
    cv.imshow('img_stackhsv',img_stack2)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    
    if sys.argv[1] == 'v':
        print('initializing video processing')
        frame_list = process_video(sys.argv[2])
        show_frames(frame_list)
    elif sys.argv[1] == 'i':
        print("getting blue")
        get_blue(sys.argv[2])
        
        #print('initializing img processing')
        #img = process_image(sys.argv[2])
        # show_img(img)
        ####
        #kernel = np.ones((5, 5))
        #img_dil = cv.dilate(img, kernel, iterations=1)
        # show_img(img_dil)
        ###
        #print('generating contours')
        #img_contours,contours = get_img_contour(img_dil)
        # show_img(img_contours)
        
        #img_stack = stack_images(0.8,([img, img_dil, img_contours]))
        #calculate_perimeter(contours)
        #measure_contours(img_contours, contours)
