from email.mime import image
from operator import truediv
from sre_constants import GROUPREF_EXISTS
import cv2 as cv
import numpy as np
from stack_images import stack_images

path = "C:/Users/fedeb/Downloads/CASTRO, ALEX 12.jpg"

def resize_frame(frame, scale=0.75):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

def get_area_of_interest(contours):

    mask = np.zeros(img.shape, dtype = 'uint8')

    cv.drawContours(mask, contours, -1, 255, cv.FILLED)

    result = cv.bitwise_and(img, img, mask = mask)

    return result, mask

img = cv.imread(path)

def empty(a):
    pass

cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 640, 240)
cv.createTrackbar("Threshold1", "Parameters", 0, 255, empty)
cv.createTrackbar("Threshold2", "Parameters", 0, 255, empty)
cv.createTrackbar("Area", "Parameters", 1000, 30000, empty)
cv.createTrackbar("Contrast", "Parameters", 1, 2, empty)



def get_grays(img):
        
    # It converts the BGR color space of image to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Generally, hue is measured in 360 degrees of a colour circle, but in OpenCV the hue scale is only 180 degrees
    # The saturation value is usually measured from 0 to 100% but in OpenCV the scale for saturation is from 0 to 255.
    # value (brightness) is usually measured from 0 to 100% but in OpenCV the scale for value is from 0 to 255
    lower_gray = np.array([0,0,0])
    upper_gray = np.array([180,64,255])

    #create a mask for gray colour using inRange function
    mask = cv.inRange(hsv, lower_gray, upper_gray)

    #perform bitwise and on the original image arrays using the mask
    resulthsv = cv.bitwise_and(img, img, mask = mask)

    gray = cv.cvtColor(resulthsv, cv.COLOR_BGR2GRAY)

    return gray


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

while True:

    gray = get_grays(img)

    img_contour_non_filtered = img.copy()
    img_contour_filtered = img.copy()

    img_blur = cv.GaussianBlur(gray, (47, 47), 1)

    threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")
    img_canny = cv.Canny(img_blur, threshold1, threshold2)

    kernel = np.ones((5, 5))
    img_dil = cv.dilate(img_canny, kernel, iterations=1)

    get_contours(img_dil, img_contour_non_filtered, False)
    get_contours(img_dil, img_contour_filtered, True)

    area_of_interest, mask = get_area_of_interest(img_contour_filtered)

    img_stack = stack_images(0.7,([img, img_blur, img_canny],
                                  [img_dil, img_contour_non_filtered,img_contour_filtered],
                                  [mask, mask, area_of_interest]))

    cv.imshow("Result", img_stack)

    if cv.waitKey(40) & 0xFF==ord('d'):
        break

cv.destroyAllWindows()