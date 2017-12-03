"""
summary: contours-based regions proposals
author: Aleksei Samarin
"""


import cv2 #-----------------------------------------------4 basic image processing
import numpy as np #---------------------------------------4 basic numeric processing


MIN_OBJECT_HEIGHT_RATIO = 0.15 #---------------------------minimal roi/image height ratio
MAX_OBJECT_HEIGHT_RATIO = 0.3 #----------------------------maximal roi/image height ratio
MIN_OBJECT_WIDTH_RATIO = 0.15 #----------------------------minimal roi/image width ratio
MAX_OBJECT_WIDTH_RATIO = 0.3 #-----------------------------maximal roi/image width ratio

CANNY_LOW_THRESHOLD_VALUE = 100 #--------------------------low threshold value for Canny filtering
CANNY_HIGH_THRESHOLD_VALUE = 200 #-------------------------high threshold value for Canny filtering


def generate_ROI_proposals_contours(image):
    """
    Performs class-independent ROI proposals based on edge detection
    :param image: numpy array - input image blob
    :return: list of int - ROI proposals: [(top left x, top left y, width, height)]
    """

    #retrieve shape of initial image
    image_height, image_width = image.shape[:2]

    #detect edges using Canny filtering
    edges = cv2.Canny(image, CANNY_LOW_THRESHOLD_VALUE, CANNY_HIGH_THRESHOLD_VALUE)

    #retrieve contours from filtered image
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # generate ROI list using found contours bounding boxes and size filtering
    roi_list = [cv2.boundingRect(contour) for contour in contours \
                if image_width * MIN_OBJECT_WIDTH_RATIO <= contour[2]
                and image_width * MAX_OBJECT_WIDTH_RATIO >= contour[2]
                and image_height * MIN_OBJECT_HEIGHT_RATIO <= contour[3]
                and image_height * MAX_OBJECT_HEIGHT_RATIO >= contour[3]]
    return roi_list
