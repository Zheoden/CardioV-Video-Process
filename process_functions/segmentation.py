import io
import cv2 as cv
import numpy as np
import tensorflow as tf

def get_ventricle_mask(img, loaded_model):
    """Creates a mask using the AI model

    Args:
        img (OpenCV Image): Image to get the mask
    """

    #img = io.imread(img)
    #img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    # Normalizing the image
    img = img * 1. / 255.

    # Reshaping the image
    img = cv.resize(img, (256, 256))
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

    # - Calls function to get mask -
    #mask = cv.imread(img)
    return predict
