from email.mime import image
from operator import truediv
from sre_constants import GROUPREF_EXISTS
import cv2 as cv
import numpy as np
from stack_images import stack_images

path = "C:/Users/fedeb/Downloads/EchoNet-Dynamic/Videos/0X1A3E7BF1DFB132FB.avi"

def resize_frame(frame, scale=0.75):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

vcap = cv.VideoCapture(path)
video_frames = []
success, image = vcap.read()
while success:
    success, frame = vcap.read()
    if not success:
        print("End of video")
        break
    video_frames.append(resize_frame(frame, 2))
vcap.release()

counter = 1
total = len(video_frames)

def empty(a):
    pass

cv.namedWindow("Parameters")
cv.resizeWindow("Parameters", 640, 240)
cv.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
cv.createTrackbar("Threshold2", "Parameters", 255, 255, empty)
cv.createTrackbar("Area", "Parameters", 1000, 30000, empty)
cv.createTrackbar("Contrast", "Parameters", 1, 2, empty)


def get_contours(img, img_contour, filter_contours):

    # RETR (Retrival method): Hay 2 metodos principales, External, que devuleve unicamente los contornos extremamente externos. Tree devuelve todos los contornos
    # CHAIN_APROX: Es la aproximacion. Con NONE obtenemos todos los puntos de contornos, y con SIMPLE obtenemos menos puntos.
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    area_min = cv.getTrackbarPos("Area", "Parameters")

    for cnt in contours:
        area = cv.contourArea(cnt)        
        if not filter_contours or area > area_min:
            cv.drawContours(img_contour, cnt, -1, (255,0,255), 7)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            x, y, w, h = cv.boundingRect(approx)
            # cv.rectangle(img_contour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            # cv.putText(img_contour, "Ps: " + str(len(approx)), (x + w + 20, y + 20), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            # cv.putText(img_contour, "Aa: " + str(int(area)), (x + w + 20, y + 45), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

while True:
    img = video_frames[counter % total]
    # original_img.convertTo(contrast, -1, 2, 0);//changing contrast//
    img_contour_non_filtered = img.copy()
    img_contour_filtered = img.copy()

    img_blur = cv.GaussianBlur(img, (47, 47), 1)

    threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")
    img_canny = cv.Canny(img_blur, threshold1, threshold2)

    kernel = np.ones((5, 5))
    img_dil = cv.dilate(img_canny, kernel, iterations=1)

    get_contours(img_dil, img_contour_non_filtered, False)
    get_contours(img_dil, img_contour_filtered, True)

    img_stack = stack_images(0.8,([img, img_blur, img_canny],
                                  [img_dil, img_contour_non_filtered,img_contour_filtered]))
    cv.imshow("Result", img_stack)
    if cv.waitKey(40) & 0xFF==ord('d'):
        break

    counter += 1

cv.destroyAllWindows()