from logging import raiseExceptions
import cv2 as cv
import numpy as np
import urllib
from process_functions import image_functions as imf, segmentation
from process_functions import segmentation as sg
from process_functions import plotter as plt
from process_functions import model_loader as ml

_CENTIMETERS = 0.2
_ERROR_VALUE = -1
MASK = 'nothing'
TMP_DIR = '/tmp/'

def process_video(url, file, model, original_scale = 1, show_images= False):
    
    try:
        vid = cv.VideoCapture(url)
        video_frames = []
        mask_list = []
        success, frame = vid.read()
        frames = 0
        
        while success:
            try:
                mask = segmentation.get_ventricle_mask(frame, model)
                try:
                    scale = imf.rescale(frame, original_scale)
                except:
                    print("Unable to get the scale, setting to default: 1")
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
        list_perimeter = []
        list_area = []
        list_muscle_t = []
        for img, mask in zip(video_frames, mask_list):
            # imf.show_img(mask, "mask")
            try:
                list_volume.append(round(imf.simpson_method(mask, scale),2))
            except Exception as error:
                print(f"Error {error} while trying to retreive ventricle volume")
                list_volume.append(_ERROR_VALUE)
            
            try:
                list_perimeter.append(round(imf.calculate_perimeter(mask, scale), 2))
                # list_perimeter.append(_ERROR_VALUE)
            except Exception as error:
                print(f"Error {error} while trying to retreive ventricle perimeter")
                list_perimeter.append(_ERROR_VALUE)
            
            try:
                list_area.append(round(imf.estimate_ventricle_area(mask, scale), 2))
                # list_area.append(_ERROR_VALUE)
            except Exception as error:
                print(f"Error {error} while trying to retreive ventricle area")
                list_area.append(_ERROR_VALUE)
            
            try:
                list_muscle_t.append(round(imf.estimate_muscle_thickness(img, mask, scale), 2))
                # list_muscle_t.append(_ERROR_VALUE)
            except Exception as error:
                print(f"Error {error} while trying to retreive muscle thickness")
                list_muscle_t.append(_ERROR_VALUE)
        
        try:
            video = imf.make_video(video_frames, file, mask_list, list_volume, TMP_DIR)
            video_name = video.split('/')[-1]
        except Exception as error:
            print(f"ERROR: {error}")
            video = 'ERROR'
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
        
        try:
            EF = imf.calculate_ef(list_volume)
        except Exception as error:
            print(f"Error {error} while trying to calculate ejection fraction")
            EF = _ERROR_VALUE    

        media = [] 
        media.append((video_name,'Video con máscaras'))
        media.append((dias_name,'Máscara de diástole'))
        media.append((sys_name,'Máscara de sístole'))

        data_set = {"ventricle_volume": list_volume, 
                    "ventricle_perimeter": list_perimeter, 
                    "ventricle_area": list_area, 
                    "muscle_thickness": list_muscle_t,
                    "media": media,
                    "ejection_fraction": EF}
        
    except Exception as error:
        print(f"An excepetion {error} was raised")

        data_set = {"ventricle_volume": _ERROR_VALUE, 
                    "ventricle_perimeter": _ERROR_VALUE, 
                    "ventricle_area": _ERROR_VALUE, 
                    "muscle_thickness": _ERROR_VALUE,
                    "media": _ERROR_VALUE,
                    "ejection_fraction": _ERROR_VALUE}
        
    #imf.show_ordered_frames(mask_list,list_volume)
        
    return data_set, video, dias, sys
    

def process_image(url, file, model, original_scale = 1, show_images = False):
    
    print('Initializing img processing for URL:')
    print(url)
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img_to_process = cv.imdecode(image, cv.IMREAD_COLOR)
    list_volume = []
    list_perimeter = []
    list_area = []
    list_muscle_t = []
    
    try:
        mask = sg.get_ventricle_mask(img_to_process, model) # replace with img when segmentation is finished
        # img_to_process = imf.make_square(img_to_process)
    except:
        raise Exception("Unable to get the mask, aborting")
    
    try:
        scale = imf.rescale(img_to_process, original_scale)
    except:
        print("Unable to get the scale, setting to default: 1")
        scale = 1

    img_to_process = imf.make_square(img_to_process)

    try:
        list_volume.append(round(imf.simpson_method(mask, scale),2))
        # volume = _ERROR_VALUE
    except Exception as error:
        print(f"Error {error} while trying to retreive ventricle volume")
        list_volume.append(_ERROR_VALUE)
    
    try:
        list_perimeter.append(round(imf.calculate_perimeter(mask, scale), 2))
    except Exception as error:
        print(f"Error {error} while trying to retreive ventricle perimeter")
        list_perimeter.append(_ERROR_VALUE)
    
    try:
        list_area.append(round(imf.estimate_ventricle_area(mask, scale), 2))
        # ventricle_area = _ERROR_VALUE
    except Exception as error:
        print(f"Error {error} while trying to retreive ventricle area")
        list_area.append(_ERROR_VALUE)
    
    try:
        list_muscle_t.append(round(imf.estimate_muscle_thickness(img_to_process, mask, scale), 2))
        # muscle_thickness = _ERROR_VALUE
    except Exception as error:
        print(f"Error {error} while trying to retreive muscle thickness")
        list_muscle_t.append(_ERROR_VALUE)

    try:
        concat_path = imf.concat_and_write(img_to_process, mask, file, TMP_DIR)
        img_name = concat_path.split('/')[-1]
    except Exception as error:
        print(f"ERROR: {error}")
        concat_path = 'ERROR'
        img_name = 'ERROR'

    media = [] 
    media.append((img_name,'Máscara'))
    
    FE = 0

    data_set = {"ventricle_volume": list_volume, 
                "ventricle_perimeter": list_perimeter, 
                "ventricle_area": list_area, 
                "muscle_thickness": list_muscle_t,
                "media": media,
                "ejection_fraction": FE}
    
    return data_set, concat_path