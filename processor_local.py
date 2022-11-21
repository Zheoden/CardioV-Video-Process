from logging import raiseExceptions
import cv2 as cv
import numpy as np
import urllib
from process_functions import image_functions as imf, segmentation
from process_functions import segmentation as sg
from process_functions import plotter as plt
from process_functions import model_loader as ml


_CENTIMETERS = 1
_ERROR_VALUE = -1
_VOLUME_LIST = [180, 190, 200, 201, 202, 199, 182]
MASK = "C:/Users/matia/cardiov/mask_test1.png"
TMP_DIR = 'D:/'

def process_video(path, file, model, original_scale = 1, show_images= False):
    
    try:
        vid = cv.VideoCapture(path)
        video_frames = []
        mask_list = []
        success, frame = vid.read()
        frames = 0
        
        while success:
            try:
                mask = segmentation.get_ventricle_mask(frame, model)
                
                try:
                    scale = imf.rescale(frame, original_scale)
                except Exception as e:
                    print(f"Unable to get the scale, setting to default: 1. Error {e}")
                    scale = 1
                    
                frame = imf.make_square(frame)
            except:
                mask = _ERROR_VALUE
            
            mask_list.append(mask)    
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
        for img, mask in zip(video_frames, mask_list):
            # imf.show_img(mask, "mask")
            try:
                list_volume.append(round(imf.simpson_method(mask, scale),2))
                # list_volume.append(_ERROR_VALUE)
            except Exception as error:
                print(f"Error {error} while trying to retreive ventricle volume")
                list_volume.append(_ERROR_VALUE)
            
            try:
                list_area1.append(imf.calculate_perimeter(img, scale))
                # list_area1.append(_ERROR_VALUE)
            except Exception as error:
                print(f"Error {error} while trying to retreive atrium area")
                list_area1.append(_ERROR_VALUE)
            
            try:
                list_area2.append(imf.estimate_ventricle_area(mask, scale))
                # list_area2.append(_ERROR_VALUE)
            except Exception as error:
                print(f"Error {error} while trying to retreive ventricle area")
                list_area2.append(_ERROR_VALUE)
            
            try:
                list_muscle_t.append(imf.estimate_muscle_thickness(img, mask, scale))
                # list_muscle_t.append(_ERROR_VALUE)
            except Exception as error:
                print(f"Error {error} while trying to retreive muscle thickness")
                list_muscle_t.append(_ERROR_VALUE)
        
        try:
            video = imf.make_video(video_frames, file, mask_list, list_volume, TMP_DIR)
            video_name = video.split('/')[-1]
        except Exception as error:
            print(f"ERROR: {error}")
            video = _ERROR_VALUE
            video_name = 'ERROR'
            
        try:
            dias, sys = imf.get_max_vol_img(video_frames, file, mask_list, list_volume, TMP_DIR)
            dias_name = dias.split('/')[-1]
            sys_name = sys.split('/')[-1]
        except Exception as error:
            print(f"ERROR: {error}")
            dias = 'ERROR'
            sys = 'ERROR'
            dias_name = 'ERROR'
            sys_name = 'ERROR'
                
        data_set = {"ventricle_volume": list_volume, 
                    "atrium_area": list_area1, 
                    "ventricle_area": list_area2, 
                    "muscle_thickness": list_muscle_t,
                    "video_name": video_name,
                    "img_1_name": dias_name,
                    "img_2_name": sys_name}
        
    except Exception as error:
        print(f"An excepetion {error} was raised")
        data_set = {"ventricle_volume": _ERROR_VALUE, 
                "atrium_area": _ERROR_VALUE, 
                "ventricle_area": _ERROR_VALUE, 
                "muscle_thickness": _ERROR_VALUE,
                "video_name": _ERROR_VALUE,
                "img_1_name": _ERROR_VALUE,
                "img_2_name": _ERROR_VALUE}
        
        
        
    #imf.show_ordered_frames(mask_list,list_volume)
    return data_set, video, dias, sys
    

def process_image(path, file, model, original_scale = 1, show_images = False):
    
    print('Initializing img processing for path:')
    print(path)
    img_to_process = cv.imread(path)
    
    try:
        mask = sg.get_ventricle_mask(img_to_process, model) # replace with img when segmentation is finished
        #mask = img_to_process
        img_to_process = imf.make_square(img_to_process)
    except:
        raise Exception("Unable to get the mask, aborting")
    
    try:
        scale = imf.rescale(img_to_process, original_scale)
    except:
        print("Unable to get the scale, setting to default: 1")
        scale = 1
    
    #imf.show_img(mask, "result!")

    try:
        volume = round(imf.simpson_method(mask, scale),2)
        # volume = _ERROR_VALUE
    except Exception as error:
        print(f"Error {error} while trying to retreive ventricle volume")
        volume = _ERROR_VALUE
    
    try:
        #atrium_area = imf.calculate_perimeter(mask, scale)
        atrium_area = _ERROR_VALUE
    except Exception as error:
        print(f"Error {error} while trying to retreive atrium area")
        atrium_area = _ERROR_VALUE
    
    try:
        #ventricle_area = imf.estimate_ventricle_area(mask, scale)
        ventricle_area = _ERROR_VALUE
    except Exception as error:
        print(f"Error {error} while trying to retreive ventricle area")
        ventricle_area = _ERROR_VALUE
    
    try:
        #muscle_thickness = imf.estimate_muscle_thickness(img_to_process, mask, scale)
        muscle_thickness = _ERROR_VALUE
    except Exception as error:
        print(f"Error {error} while trying to retreive muscle thickness")
        muscle_thickness = _ERROR_VALUE
        
    try:
        concat_path = imf.concat_and_write(img_to_process, mask, file, TMP_DIR)
        concat_name = concat_path.split('/')[-1]
    except Exception as error:
        print(f"ERROR: {error}")
        concat_path = 'ERROR'
        concat_name = 'ERROR'
    
    data_set = {"ventricle_volume": volume, 
                "atrium_area": atrium_area, 
                "ventricle_area": ventricle_area, 
                "muscle_thickness": muscle_thickness,
                "video_name": 'None',
                "img_1_name": concat_name,
                "img_2_name": 'None'}
    
    return data_set, concat_path


def make_a_graph():
    volume_list = _VOLUME_LIST
    plt.create_graph(_VOLUME_LIST, 'Frame', 'Volume - CM3', 'Variación de volumen')