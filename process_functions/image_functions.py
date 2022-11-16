import cv2 as cv
import numpy as np
import cv2 as cv
import math
from PIL import Image

def rescale(image, scale):

    (h, w) = image.shape[:2]
    if( h > w ):
        p = 256 / h
    else:
        p = 256 / w
    
    return scale * p

def make_square(img, min_size=256, new_size=(256,256)):
    x, y, z = img.shape
    size = max(min_size, x, y)
    new_img = np.zeros((size,size,3), dtype='uint8')
    x_offset = int((size - x) / 2)
    y_offset = int((size - y) / 2)
    new_img[x_offset:size-x_offset, y_offset:size-y_offset] = img
    return cv.resize(new_img, new_size)

# def multiple_make_square(imgs_path, output_path):
#     imgs_list = os.listdir(files_path)
#     for img in imgs_list:
#         img_basename = os.path.basename(img)
#         input_file = files_path + '/' + img
#         out_path = output_path + '/' + img_basename
#         img = Image.open(input_file)
#         new_img = make_square(img)
#         print(f'Final img size: {new_img.height}x{new_img.width}')
#         new_img.save(out_path)

def resize_frame(frame, scale=0.75):
    """_summary_

    Args:
        frame (opencv image): image to resize
        scale (float, optional): scale for resizing. Defaults to 0.75.

    Returns:
        opencv image: image resized
    """

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

def get_area_of_interest(img, show_images= False):
    """Function that process an image and returns a mask with the desired area

    Args:
        img (open cv image): image to process
        show_images (bool, optional): Defaults to True.

    Returns:
        OpenCV image: mask
    """
    
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

    #show_img(mask, "Mask") if show_images else 1

    return mask

def get_grays(img):
    """Converts image to gray scale using HSV methods

    Args:
        img (open cv image): image to process

    Returns:
        OpenCV image: processed image
    """
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

def get_img_borders_no_dil(img, threshold = (3,3), canny1= 125, canny2= 175):
    """Function that returns a processed image, showing only its edges with no dilatation 

    Args:
        path (_type_): image path
    """
    # Blurring may be necessary to not have extra edges
    blur = cv.GaussianBlur(img, threshold, cv.BORDER_DEFAULT)
    # Canny edge detecting:
    edges = cv.Canny(blur, canny1, canny2)
    
    return edges

def get_img_max_contours(img, desired_amount):
    """
    Function that returns a list of masks generated from the desired amount of max_area contours. It also returns a list with corresponding contours.
    """
    contours, hierarchies = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print('finding countours')
    color = (255,0,255)
    contour_thickness = 1

    #print('writing contours')
    #print(f'{len(contours)} contours found')
    
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

def get_img_contour(img):
    """Process an image to get its contours

    Args:
        img (OpenCV image): Image to process

    Returns:
        OpenCV Image, contours: Image with contours drawn, list of contours
    """
    
    # RETR (Retrival method): Hay 2 metodos principales, External, que devuleve unicamente los contornos extremamente externos. Tree devuelve todos los contornos
    # CHAIN_APROX: Es la aproximacion. Con NONE obtenemos todos los puntos de contornos, y con SIMPLE obtenemos menos puntos.
    #CHAIN_APPROX_NONE test
    
    contours, hierarchies = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #area_min = 1000
    #print('finding countours')
    color = (255,0,255)
    contour_thickness = 2
    
    #print('creating blank img')
    blank = np.zeros(img.shape, dtype = 'uint8')
    #print('writing contours')
    #print(f'{len(contours)} contours found')
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        #if area > area_min:
        cv.drawContours(blank, [cnt], -1, color, contour_thickness)
            
    return blank, contours

def get_img_contour_max_area(img):
    """Process an image to get its max area contour

    Args:
        img (OpenCV image): Image to process

    Returns:
        OpenCV Image, contours: Image with contours drawn, list of contours
    """
    
    contours, hierarchies = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print('finding countours')
    color = (255,0,255)
    contour_thickness = 1
    
    #print('creating blank img')
    blank = np.zeros(img.shape, dtype = 'uint8')
    #print('writing contours')
    #print(f'{len(contours)} contours found')
    
    areas = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        areas.append(area)
    max_area = max(areas)
    max_area_index = areas.index(max_area)

    max_contour = contours[max_area_index]
    cv.drawContours(blank, [max_contour], -1, color, contour_thickness)
    
    return blank, max_contour

def get_minimum_area(img, name_value = 'value'):
    """Process an image to get its minimum border rectangle

    Args:
        img (OpenCV image): Image to process

    Returns:
        OpenCV Image: Image cropped to its minimum rectangle
    """
    
    img_borders = get_img_borders_no_dil(img)
    img_contours, max_contour = get_img_contour_max_area(img_borders)
    #show_img(img_contours, "CONTOUR")
    x, y, w, h = cv.boundingRect(max_contour)
    cropped = img[y:y+h,x:x+w]

    return cropped

def simpson_method(img, scale, cuts = 3):
    """Calculates volume using Simpson Method, that consists in cropping the image in multiple rectangles and asume that are perfect cilinders

    Args:
        img (OpenCV Image): Image to process

    Returns:
        Float: Final volume 
    """
    
    cropped = get_minimum_area(img, 'original')
    (h, w) = cropped.shape[:2]
    evaluate_height = cuts
    counter = evaluate_height
    actual_height = 0
    contador = 1
    img_cut_list = []
    
    # dynamic cut -> deprecated
    # while( counter <= h ):
    #     img_cut = np.zeros((evaluate_height+2,w+2,3),dtype="uint8")
    #     img_cut[1:1+evaluate_height,1:1+w] = cropped[actual_height:counter,0:w]
    #     cropped_min = get_minimum_area(img_cut)
    #     show_img(cropped_min)
    #     img_cut_list.append(cropped_min)
    #     actual_height = counter
    #     counter += evaluate_height
    #     contador += 1

    while( counter <= h ):
        #Si el resultado entre la altura máxima y el próximo corte es menor al contador fijo para cortar, se suman las cantidades y se corta para el resto total
        if( 1 < h - actual_height % counter < 2 ):
            counter = h
        img_cut = np.zeros((evaluate_height+2,w+2,3),dtype="uint8")
        img_cut[1:1+evaluate_height,1:1+w] = cropped[actual_height:counter,0:w]
        cropped_min = get_minimum_area(img_cut)
        #show_img(cropped_min)
        img_cut_list.append(cropped_min)
        actual_height = counter
        counter += evaluate_height
        contador += 1
    
    final_volume = calculate_cilinder_volume(img_cut_list, scale)
    print(f"Ventricle volume: {final_volume} cm3")
    return final_volume
    
def calculate_cilinder_volume(list_cilinders, scale):
    """Calculates the volume of a cilinder

    Args:
        list_cilinders (list): list of cilinders to measure
        centimeters (int): scale -> pixels to cm 

    Returns:
        Float: total volume 
    """
    counter = 0
    total_volume = 0
    for cil in list_cilinders:
        (h, w) = cil.shape[:2]
        height = (h - 2) * scale
        width = (w - 2) * scale
        total_volume = total_volume + height*(width/2)**2*math.pi
        counter += 1
    
    return total_volume

def get_left_ventricle_walls(img, mask):
    """Applies the mask to the image and cuts it with a margin applied to the bounding rectangle from the mask

    Args:
        img (OpenCV Image): Image to process
        mask (OpenCV Image): Mask for the image to process

    Returns:
        OpenCV Image: Cropped image to get the ventricle walls
    """
    
    x_margin = 10
    y_margin = 0

    #print("INVERTED MASK")
    inverted_mask = cv.bitwise_not(mask)
    
    #getting cropping rectangle
    
   # print("GETTING BORDERS")
    mask_borders = get_img_borders(mask)
    
    #print("GETTING CONTOURS")
    mask_contours, max_contours = get_img_max_contours(mask_borders, 1)
    x, y, w, h = cv.boundingRect(max_contours[0])

    #applying mask to image
    
    #print("MASKING")
    masked_img = cv.bitwise_and(img, img, mask = inverted_mask)
    #show_img(mask)
    #show_img(masked_img,'final')

    #preventing margin overflow
    (height, width) = mask.shape[:2]
    low_y = max(y-y_margin, 0)
    high_y = min(y+h+y_margin, height)
    low_x = max(x-x_margin, 0)
    high_x = min(x+w+x_margin, width)
    
    #print("CROPPING")
    cropped_img = masked_img[low_y:high_y,low_x:high_x]

    return cropped_img

def get_minimum_areas(img, desired_amount):
    """Process image to get its minimum areas

    Args:
        img (OpenCV Image): Image to process
        desired_amount (_type_): Number of times to crop the image

    Returns:
        List: list of cropped images
    """

    img_borders = get_img_borders_no_dil(img)
    img_contours, max_contours = get_img_max_contours(img_borders, desired_amount)

    list_of_cropped_images = []
    for i in range(len(max_contours)):        
        x, y, w, h = cv.boundingRect(max_contours[i])
        cropped = img[y:y+h,x:x+w]
        list_of_cropped_images.append(cropped)

    return list_of_cropped_images

def calculate_wall_thickness(wall_cuts_list, scale):
    """Calculates the wall thickness of a ventricle

    Args:
        wall_cuts_list (List): list of OpenCV images to measure
        centimeters (Int): Scale -> pixels to cm

    Returns:
        Float: Average muscle thickness
    """
    amount_of_cuts = len(wall_cuts_list)
    thickness_sum = 0

    i = 0
    for wall_cut in wall_cuts_list:
        i += 1
        (h, w) = wall_cut.shape[:2]
        #print(f'wall_cut pixel width: {w}')
        # TODO: Fede, Probar y agregar filtro de thickness altos y bajos
        width = (w - 2) * scale
        #print(f"Calculating average wall thickness with thickness of : {width} centimeters")
        thickness_sum += width
    
    average_thickness = thickness_sum / amount_of_cuts

    return average_thickness

def estimate_muscle_thickness(img, mask_in = 'nothing', scale = 1, show_images = False):
    """Estimates the thickness of the ventricle walls.

    Args:
        img (OpenCV Image): Image to process
        mask (OpenCV Image): Mask for the image to process
        show_images (bool, optional): Defaults to False.

    Raises:
        Exception: No mask was granted
        
    Returns:
        Float: Average muscle thickness
    """
    
    if mask_in == 'nothing':
        raise Exception("No mask was granted") 
    
    # mask = cv.cvtColor(mask_in, cv.COLOR_BGR2GRAY)
    mask = mask_in
    print("GETTING WALLS")
    walls_img = get_left_ventricle_walls(img, mask)

    #show_img(walls_img, "walls_img")# if show_images else -1

    (h, w) = walls_img.shape[:2]
    
    #print(f"width:{w} height:{h}")
    evaluate_height = h // 8
    #print(f"ev height:{evaluate_height}")
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
    
    final_wall_thickness = calculate_wall_thickness(img_cut_list, scale)
        
    return final_wall_thickness
    
#TODO Funciona un poquito mal
def calculate_perimeter(img, scale):
   
    # img = get_grays(img)
    cropped = get_minimum_area(img, 'original')
    mask_borders = get_img_borders_no_dil(cropped)
    cropped_conts, contours = get_img_contour(mask_borders) 
    #img_contours, max_contour = get_img_contour_max_area(mask_borders)
    #show_img(cropped_conts,"no max")
    #show_img(img_contours, "max area")
  
    perimeter_list = []
    for item in contours:
        perimeter_list.append(cv.arcLength(item,True))
    
    perimeter = max(perimeter_list)
    print(f"PERIMETER = {perimeter}")
    return perimeter

def estimate_ventricle_area(img, scale):
    """Estimates the area of an object

    Args:
        img (OpenCV image): image to estimate area
    """
    cropped = get_minimum_area(img, 'original')
    mask_borders = get_img_borders_no_dil(cropped)
    cropped_conts, contours = get_img_contour(mask_borders)    
    # TODO: Fede
    area = cv.contourArea(contours[0]) * (scale)**2
    return area
    
def show_img(img, name = 'Heart'):
    """Show image wrapper

    Args:
        img (OpenCV image): image to show
        name (str, optional): Name. Defaults to 'Heart'.
    """
    #print(f'Showing {name} image')
    cv.imshow(name, img)
    cv.waitKey(0)
    
def make_mask_video(img, mask):
    
    inverted_mask = cv.bitwise_not(mask)
    
    #getting cropping rectangle
    
   # print("GETTING BORDERS")
    mask_borders = get_img_borders(mask)
    
    #print("GETTING CONTOURS")
    mask_contours, max_contours = get_img_max_contours(mask_borders, 1)
    x, y, w, h = cv.boundingRect(max_contours[0])

    #applying mask to image
    
    #print("MASKING")
    masked_img = cv.bitwise_and(img, img, mask = inverted_mask)
    #show_img(masked_img)
    
def show_ordered_frames(list_frames, list_names):

    counter = 1
    cont = 0
    total = len(list_frames)
    tot = len(list_names) - 1
    # resized_list = map(lambda frame: resize_frame(frame, 2), list)
    resized_list = []
    for frame in list_frames:
        resized_list.append(resize_frame(frame, 2))
    while True and cont < 150:
        #print(f'Showing img {counter % total} of {total}')
        if(list_names[cont%tot] > 100):
            cv.imshow(f'Mask',resized_list[counter % total])
            print(f"VOLUME FRAME: {str(round(list_names[cont%tot],2))} cm3")
            if cv.waitKey(100) & 0xFF==ord('d'):
                break
        
        counter += 1
        cont += 1 

    cv.destroyAllWindows()
