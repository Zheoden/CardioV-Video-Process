from logging import raiseExceptions
import cv2 as cv
import numpy as np
import urllib
from process_functions import image_functions as imf, segmentation
from process_functions import segmentation as sg

_CENTIMETERS = 0.2
_ERROR_VALUE = -1
MASK = 'nothing'

def process_video(url, show_images= False):
    
    try:
        vid = cv.VideoCapture(url)
        video_frames = []
        success, frame = vid.read()
        frames = 0
        
        while success or frames <= 40:
            
            #
            # CALL AI METHOD HERE - frame_mask = aimethod(frame) -> and pass it to both volume and wall thickness methods
            #
            
            video_frames.append(frame)
            success, frame = vid.read()
            if not success:
                print('End of video')
                break
            frames += 1
        vid.release()
        
        list_volume = []
        list_area1 = []
        list_area2 = []
        list_muscle_t = []
        for img in video_frames:
            
            try:
                list_volume.append(imf.simpson_method(img))
            except Exception as error:
                print(f"Error {error} while trying to retreive ventricle volume")
                list_volume.append(_ERROR_VALUE)
            
            try:
                list_area1.append(imf.estimate_atrium_area(img))
            except Exception as error:
                print(f"Error {error} while trying to retreive atrium area")
                list_area1.append(_ERROR_VALUE)
            
            try:
                list_area2.append(imf.estimate_ventricle_area(img))
            except Exception as error:
                print(f"Error {error} while trying to retreive ventricle area")
                list_area2.append(_ERROR_VALUE)
            
            try:
                list_muscle_t.append(imf.estimate_muscle_thickness(img, MASK))
            except Exception as error:
                print(f"Error {error} while trying to retreive muscle thickness")
                list_muscle_t.append(_ERROR_VALUE)
        
        data_set = {"ventricle_volume": list_volume, 
                    "atrium_area": list_area1, 
                    "ventricle_area": list_area2, 
                    "muscle_thickness": list_muscle_t}
    
    except Exception as error:
        print(f"An excepetion {error} was raised")
        data_set = {"ventricle_volume": _ERROR_VALUE, 
                    "atrium_area": _ERROR_VALUE,
                    "ventricle_area": _ERROR_VALUE,
                    "muscle_thickness": _ERROR_VALUE}
        
    return data_set
    

def process_image(url, show_images = False):
    
    print('Initializing img processing for URL:')
    print(url)
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img_to_process = cv.imdecode(image, cv.IMREAD_COLOR)
    
    try:
        mask = sg.get_ventricle_mask(img_to_process) # replace with img when segmentation is finished
    except:
        raise Exception("Unable to get the mask, aborting")
    
    try:
        volume = imf.simpson_method(mask)
    except Exception as error:
        print(f"Error {error} while trying to retreive ventricle volume")
        volume = _ERROR_VALUE
    
    try:
        atrium_area = imf.calculate_perimeter(mask)
    except Exception as error:
        print(f"Error {error} while trying to retreive atrium area")
        atrium_area = _ERROR_VALUE
    
    try:
        ventricle_area = imf.estimate_ventricle_area(mask)
    except Exception as error:
        print(f"Error {error} while trying to retreive ventricle area")
        ventricle_area = _ERROR_VALUE
    
    try:
        muscle_thickness = imf.estimate_muscle_thickness(img_to_process, mask)
    except Exception as error:
        print(f"Error {error} while trying to retreive muscle thickness")
        muscle_thickness = _ERROR_VALUE
    
    data_set = {"ventricle_volume": volume, 
                "atrium_area": atrium_area, 
                "ventricle_area": ventricle_area, 
                "muscle_thickness": muscle_thickness}
    
    return data_set
