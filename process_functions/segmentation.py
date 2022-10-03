import cv2 as cv

def get_ventricle_mask(img):
    """Creates a mask using the AI model

    Args:
        img (OpenCV Image): Image to get the mask
    """
    # - Calls function to get mask -
    mask = cv.imread(img)
    return mask
