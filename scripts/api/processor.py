import cv2 as cv
#import os
import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import imutils
#from make_video import make_video_from_edge_maps
#import image_processing as pi

def resize_frame(frame, scale=0.75):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

def get_area_of_interest(img, show_images= True):
    
    gray = get_grays(img)
    img_borders = get_img_borders(gray, canny1= 0, canny2= 0)
    contours, hierarchy = cv.findContours(img_borders, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    areas = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        areas.append(area)

    max_area = max(areas)
    max_area_index = areas.index(max_area)
    area_of_interest = contours[max_area_index]
    mask = np.zeros(img_borders.shape, dtype = 'uint8')
    cv.drawContours(mask, [area_of_interest], -1, 255, -1)

    show_img(mask, "Mask") if show_images else 1

    return mask

def get_grays(img):
    
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Generally, hue is measured in 360 degrees of a colour circle, but in OpenCV the hue scale is only 180 degrees
    # The saturation value is usually measured from 0 to 100% but in OpenCV the scale for saturation is from 0 to 255.
    # value (brightness) is usually measured from 0 to 100% but in OpenCV the scale for value is from 0 to 255
    lower_gray = np.array([0,0,0])
    upper_gray = np.array([180,64,255])
    mask = cv.inRange(hsv, lower_gray, upper_gray)
    resulthsv = cv.bitwise_and(img, img, mask = mask)
    gray = cv.cvtColor(resulthsv, cv.COLOR_BGR2GRAY)

    return gray

def get_img_borders(img, threshold = (3,3), canny1= 125, canny2= 175):
    """Function that returns a processed image, showing only its edges

    Args:
        path (_type_): image path
    """
    #img_resized = resize_frame(img)
    # Blurring may be necessary to not have extra edges
    blur = cv.GaussianBlur(img, threshold, cv.BORDER_DEFAULT)
    # Canny edge detecting:
    edges = cv.Canny(blur, canny1, canny2)
    kernel = np.ones((5, 5))
    img_dil = cv.dilate(edges, kernel, iterations=1)
    
    return img_dil

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

def get_contours(img, img_contour, filter_contours):

    # RETR (Retrival method): Hay 2 metodos principales, External, que devuleve unicamente los contornos extremamente externos. Tree devuelve todos los contornos
    # CHAIN_APROX: Es la aproximacion. Con NONE obtenemos todos los puntos de contornos, y con SIMPLE obtenemos menos puntos.
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    area_min = cv.getTrackbarPos("Area", "Parameters")

    for cnt in contours:
        area = cv.contourArea(cnt)
        if not filter_contours or area > area_min:
            cv.drawContours(img_contour, cnt, -1, (255,0,255), 7)

            # peri = cv.arcLength(cnt, True)
            # approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            # x, y, w, h = cv.boundingRect(approx)
            # cv.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            # cv.putText(img_contour, "Ps: " + str(len(approx)), (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            # cv.putText(img_contour, "Aa: " + str(int(area)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    return img

def get_img_contour(img):
    """
    Args:
        img: img with dilated borders
    """
    
    # RETR (Retrival method): Hay 2 metodos principales, External, que devuleve unicamente los contornos extremamente externos. Tree devuelve todos los contornos
    # CHAIN_APROX: Es la aproximacion. Con NONE obtenemos todos los puntos de contornos, y con SIMPLE obtenemos menos puntos.
    #CHAIN_APPROX_NONE test
    
    contours, hierarchies = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    area_min = 1000
    print('finding countours')
    color = (255,0,255)
    contour_thickness = 2
    
    print('creating blank img')
    blank = np.zeros(img.shape, dtype = 'uint8')
    print('writing contours')
    print(f'{len(contours)} contours found')
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > area_min:
            cv.drawContours(blank, [cnt], -1, color, contour_thickness)
            
    return blank,contours


def show_frames(list):

    counter = 1
    total = len(list)
    for frame in list:
        print(f'Showing img {counter} of {total}')
        cv.imshow(f'Heart_{counter}',frame)
        cv.waitKey(0)
        counter += 1
        
def calculate_perimeter(cont):
   
    for item in cont:
        perimeter = cv.arcLength(item,True)
        print(f"PERIMETER = {perimeter}")

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

    
def process_selector(type, path, show_images= False):
    
    ret_val = 'hi'
    if type == 'v':
        print('initializing video processing')
        frame_list = process_video(path, show_images)
        ret_val = 'Video processed'
    elif type == 'i':
        print('initializing img processing. Path:')
        print(path)
        img_to_process = cv.imread(path)
        mask = get_area_of_interest(img_to_process, show_images)
        img = process_image(img_to_process, mask, show_images)
        ret_val = 'Image processed'
    return ret_val

# if __name__ == "__main__":
    
#     process_selector(sys.argv[1], sys.argv[2])
