from turtle import width
import cv2 as cv
#import os
import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import imutils
import math

def get_img_borders(img, threshold = (3,3), canny1= 125, canny2= 175):
    """Function that returns a processed image, showing only its edges

    Args:
        path (_type_): image path
    """
    # Blurring may be necessary to not have extra edges
    blur = cv.GaussianBlur(img, threshold, cv.BORDER_DEFAULT)
    # Canny edge detecting:
    edges = cv.Canny(blur, canny1, canny2)
    
    return edges

def get_img_contour(img):
    
    contours, hierarchies = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print('finding countours')
    color = (255,0,255)
    contour_thickness = 1
    
    print('creating blank img')
    blank = np.zeros(img.shape, dtype = 'uint8')
    print('writing contours')
    print(f'{len(contours)} contours found')
    
    areas = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        areas.append(area)
    max_area = max(areas)
    max_area_index = areas.index(max_area)

    max_contour = contours[max_area_index]
    cv.drawContours(blank, [max_contour], -1, color, contour_thickness)
    
    # for cnt in contours:
    #     cv.drawContours(blank, [cnt], -1, color, contour_thickness)
            
    return blank, max_contour


def get_minimum_area(img, name_value = 'value'):
    
    img_borders = get_img_borders(img)
    img_contours, max_contour = get_img_contour(img_borders)
    #show_img(img_contours, "CONTOUR")
    x, y, w, h = cv.boundingRect(max_contour)
    cropped = img[y:y+h,x:x+w]

    return cropped

def cut_image():
    
    img = cv.imread("C:/Users/matia/OneDrive/Escritorio/mask.jpg")
    #img = resize_frame(img, 4)
    #show_img(img)
    cropped = get_minimum_area(img, 'original')
    (h, w) = cropped.shape[:2]
    
    print(f"width:{w} height:{h}")
    evaluate_height = h // 12
    #evaluate_height = 10
    print(f"ev height:{evaluate_height}")
    counter = evaluate_height
    actual_height = 0
    img_cut_list = []
    while( counter <= h ):
        img_cut = np.zeros((evaluate_height+2,w+2,3),dtype="uint8")
        img_cut[1:1+evaluate_height,1:1+w] = cropped[actual_height:counter,0:w]
        cropped_min = get_minimum_area(img_cut)
        img_cut_list.append(cropped_min)
        actual_height = counter
        counter+=evaluate_height
    
    final_volume = calculate_volume(img_cut_list, 0.02)
    print(f"Ventricle volume: {final_volume} cm3")
    
def calculate_volume(list_cilinders, centimeters):
    
    total_volume = 0
    for cil in list_cilinders:
        (h, w) = cil.shape[:2]
        height = h*centimeters
        width = w*centimeters
        print(f"Calculating volume with height of : {height} and diameter of : {width} centimeters")
        total_volume = height*(width/2)**2*math.pi
    
    return total_volume
    
def show_img(img, name = 'Heart'):

    print(f'Showing {name} image')
    cv.imshow(name, img)
    cv.waitKey(0)

if __name__ == "__main__":
    
    cut_image()
