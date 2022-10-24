import io
import cv2 as cv
import numpy as np
import tensorflow as tf
from process_functions.image_functions import make_square, show_img

def get_ventricle_mask(img, loaded_model):
    """Creates a mask using the AI model

    Args:
        img (OpenCV Image): Image to get the mask
    """

    # Reshaping the image
    img = make_square(img)

    # Normalizing the image
    img = img * 1. / 255.
    # Converting the image into array
    img = np.array(img, dtype=np.float64)

    img = np.reshape(img, (256, 256, 3))
    # reshaping the image from 256,256,3 to 1,256,256,3
    img = np.reshape(img, (1, 256, 256, 3))

    # Creating a empty array of shape 1,256,256,1
    X = np.empty((1, 256, 256, 1))
    # standardising the image
    img -= img.mean()
    img /= img.std()
    # converting the shape of image from 256,256,3 to 1,256,256,3
    axis = 0
    # image is your tensor
    tf.expand_dims(img, axis)

    # make prediction
    predict = loaded_model.predict(img)[0]

    predict_copy = (predict*255).astype(np.uint8)
    # show_img(predict_copy,"predict_copy")

    # mask = np.asarray(predict_copy).squeeze().round()
    # show_img(mask,"mask")
    # show_img(predict.set(cv.CAP_PROP_FORMAT, cv.CV_8UC1),"predicted")
    # - Calls function to get mask -
    #mask = cv.imread(img)


    return predict_copy
