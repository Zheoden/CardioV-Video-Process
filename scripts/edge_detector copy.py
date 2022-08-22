import cv2 as cv
#import os
import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import imutils
from make_video import make_video_from_edge_maps
import get_area_of_interest as pi

def resize_frame(frame, scale=0.75):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

def get_img_borders(img, threshold = (3,3)):
    """Function that returns a processed image, showing only its edges

    Args:
        path (_type_): image path
    """

    img_resized = resize_frame(img)
    # Blurring may be necessary to not have extra edges
    blur = cv.GaussianBlur(img_resized, threshold, cv.BORDER_DEFAULT)
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
        #PRUEBAS
        frame_edges = get_img_borders(frame, (5,5))
        video_frames.append(frame_edges)
    vid.release()

    return video_frames

def process_image(path, show_images = False):
    """
        Process image first read the img from the path provided
        Then it follows these next steps:
        1- Apply color filter to get clean areas
        2- Convert img to black and white
        3- Call function get borders
        4- Call function get contours 
    """    
    img = cv.imread(path)

    show_img(img,'Heart Original') if show_images
    
    gray = pi.get_grays(img)
    show_img(gray,"Areas in black and white") if show_images
    #print("applying thrershold")
    #thresh, thresh_img = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
    #img_resized = resize_frame(gray)
    
    img_edges = get_img_borders(gray)
    show_img(img_edges,'Heart Edges') if show_images

    contours, img_processed = pi.get_contours(img_edges, img)

    show_img(img_processed,'Heart with contours') if show_images
    show_img(contours,'Only contours heart') if show_images

    return img_edges


def get_img_contour(img):

    print('finding countours')
    contours, hierarchies = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    print('creating blank img')
    blank = np.zeros(img.shape, dtype = 'uint8')
    print('writing contours')
    print(f'{len(contours)} contours found')
    # -1 is to use every contour detected
    cv.drawContours(blank, contours, -1, (255,0,255), 2)
    return blank


def show_frames(list):

    counter = 1
    total = len(list)
    for frame in list:
        print(f'Showing img {counter} of {total}')
        cv.imshow(f'Heart_{counter}',frame)
        cv.waitKey(0)
        counter += 1

#PRUEBAS
def show_frames2(list):

    counter = 1
    total = len(list)
    # resized_list = map(lambda frame: resize_frame(frame, 2), list)
    resized_list = []
    for frame in list:
        resized_list.append(resize_frame(frame, 2))
    while True:
        print(f'Showing img {counter % total} of {total}')
        cv.imshow('Video',resized_list[counter % total])
        counter += 1

        if cv.waitKey(20) & 0xFF==ord('d'):
            break

    cv.destroyAllWindows()

def show_img(img, name = 'Heart'):

    print(f'Showing {name} image')
    cv.imshow(name, img)
    cv.waitKey(0)

if __name__ == "__main__":

    if sys.argv[1] == 'v':
        print('initializing video processing')
        frame_list = process_video(sys.argv[2])
        #PRUEBAS
        # show_frames2(frame_list)
        # print("Generating video")
        make_video_from_edge_maps(frame_list, "video1.mp4")

    elif sys.argv[1] == 'i':
        print('initializing img processing')
        img = process_image(sys.argv[2])
        show_img(img)
        print('generating contours')
        img_contours = get_img_contour(img)
        show_img(img_contours)
