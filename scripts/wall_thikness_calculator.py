from subprocess import HIGH_PRIORITY_CLASS
from turtle import shape
import cv2 as cv
import numpy as np
import os

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

def show_img(img, name = 'Heart'):

    print(f'Showing {name} image')
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_img_max_contours(img, desired_amount):
    """Function that returns a list of masks generated from the desired amount of max_area contours. It also returns a list with corresponding contours.
    """
    contours, hierarchies = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print('finding countours')
    color = (255,0,255)
    contour_thickness = 1

    print('writing contours')
    print(f'{len(contours)} contours found')
    
    areas = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        areas.append(area)
    # clone the areas list by slicing
    copy_of_areas = areas[:]

    list_of_max_area_index = []
    for i in range(desired_amount):
        if(not len(copy_of_areas)):
            break
        max_area = max(copy_of_areas)
        list_of_max_area_index.append(areas.index(max_area))
        copy_of_areas.remove(max_area)

    list_of_blanks = []
    list_of_max_contours = []
    for max_area_index in list_of_max_area_index:
        blank = np.zeros(img.shape, dtype = 'uint8')
        max_contour = contours[max_area_index]
        cv.drawContours(blank, [max_contour], -1, color, contour_thickness)
        list_of_max_contours.append(max_contour)
        list_of_blanks.append(blank)
            
    return list_of_blanks, list_of_max_contours

def get_left_ventricle_walls(img_path, mask_path):
    '''A function that, from an image and a mask, applies the mask to the image and cuts it with a margin applied to the bounding rectangle from the mask
    '''

    x_margin = 10
    y_margin = 0

    img = cv.imread(img_path)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    inverted_mask = cv.bitwise_not(mask)
    
    #getting cropping rectangle
    mask_borders = get_img_borders(mask)
    mask_contours, max_contours = get_img_max_contours(mask_borders, 1)
    x, y, w, h = cv.boundingRect(max_contours[0])

    #applying mask to image
    masked_img = cv.bitwise_and(img, img, mask = inverted_mask)

    #preventing margin overflow
    (height, width) = mask.shape[:2]
    print(shape)
    low_y = max(y-y_margin, 0)
    high_y = min(y+h+y_margin, height)
    low_x = max(x-x_margin, 0)
    high_x = min(x+w+x_margin, width)

    print

    cropped_img = masked_img[low_y:high_y,low_x:high_x]

    return cropped_img

def get_minimum_areas(img, desired_amount):

    img_borders = get_img_borders(img)
    img_contours, max_contours = get_img_max_contours(img_borders, desired_amount)

    list_of_cropped_images = []
    for i in range(len(max_contours)):        
        x, y, w, h = cv.boundingRect(max_contours[i])
        cropped = img[y:y+h,x:x+w]
        list_of_cropped_images.append(cropped)

    return list_of_cropped_images

def calculate_wall_thickness(wall_cuts_list, centimeters):
    print("Wall thickness calculated")

    amount_of_cuts = len(wall_cuts_list)
    thickness_sum = 0

    i = 0
    for wall_cut in wall_cuts_list:
        i += 1
        (h, w) = wall_cut.shape[:2]
        print(f'wall_cut pixel width: {w}')
        show_img(wall_cut, f"wall_cut ( {i} / {amount_of_cuts} )")     
        width = w*centimeters
        print(f"Calculating average wall thickness with thickness of : {width} centimeters")
        thickness_sum += width
    
    average_thickness = thickness_sum / amount_of_cuts

    return average_thickness


def get_wall_thickness(img_path, mask_path):

    walls_img = get_left_ventricle_walls(img_path, mask_path)

    show_img(walls_img, "walls_img")

    (h, w) = walls_img.shape[:2]
    
    print(f"width:{w} height:{h}")
    evaluate_height = h // 12
    print(f"ev height:{evaluate_height}")
    counter = evaluate_height
    actual_height = 0
    img_cut_list = []
    while( counter <= h ):
        img_cut = np.zeros((evaluate_height+2,w+2,3),dtype="uint8")
        img_cut[1:1+evaluate_height,1:1+w] = walls_img[actual_height:counter,0:w]
        cropped_minimun_areas = get_minimum_areas(img_cut,2)
        img_cut_list.extend(cropped_minimun_areas)
        actual_height = counter
        counter+=evaluate_height
    
    final_wall_thickness = calculate_wall_thickness(img_cut_list, 0.02)
    print(f"Average wall thickness of left ventricle: {final_wall_thickness} cm")

if __name__ == "__main__":

    img_path = 'CASTRO,_ALEX_4_f0/images/CASTRO,_ALEX_4_f0.jpg'
    mask_path = 'CASTRO,_ALEX_4_f0/masks/CASTRO,_ALEX_4_f0.png'

    get_wall_thickness(img_path, mask_path)
    