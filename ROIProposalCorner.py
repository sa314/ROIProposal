"""
summary: corners-based regions proposals
author: Aleksei Samarin
"""


import cv2 #-----------------------------------------------4 basic image processing
import numpy as np #---------------------------------------4 basic numeric processing


MIN_OBJECT_HEIGHT_RATIO = 0.15 #---------------------------minimal roi/image height ratio
MAX_OBJECT_HEIGHT_RATIO = 0.3 #----------------------------maximal roi/image height ratio
MIN_OBJECT_WIDTH_RATIO = 0.15 #----------------------------minimal roi/image width ratio
MAX_OBJECT_WIDTH_RATIO = 0.3 #-----------------------------maximal roi/image width ratio
SCANNING_WINDOW_STEP_RATIO_WIDTH = 0.1 #-------------------roi generation scanning window step ratio width
SCANNING_WINDOW_STEP_RATIO_HEIGHT = 0.1 #------------------roi generation scanning window step ratio height

MIN_HIT_ROI_CORNERS_NUMBER = 4 #---------------------------minimal corners per roi hit number
DETECTED_ROI_NUMBER = 10000 #------------------------------corners detections number
MINIMAL_APPROPRIATE_CORNERS_CONFIDENCE = 0.01 #------------minimal appropriate detected corners confidence
MINIMAL_CORNERS_DISTANCE_PX = 10 #-------------------------minimal distance between corners in pixels


def generate_ROI_proposals_corners_hit_quantity(image):
    """
    Performs class-independent ROI proposals based on hit corners number
    :param image: numpy array - input image blob
    :return: list of int - ROI proposals: [(top left x, top left y, width, height)]
    """

    #retrieve shape of initial image
    image_height, image_width = image.shape[:2]

    #retrieve float grayscale image
    grayscale_image = np.float32(image) if len(image.shape) == 2 \
                      else np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    #retrieve corners positions from grayscale image
    corners = [corner_info.ravel() for corner_info in
                    np.int0(cv2.goodFeaturesToTrack(grayscale_image,
                                                    DETECTED_ROI_NUMBER,
                                                    MINIMAL_APPROPRIATE_CORNERS_CONFIDENCE,
                                                    MINIMAL_CORNERS_DISTANCE_PX))]

    #generate ROI list each of them contains at least MIN_HIT_ROI_CORNERS_NUMBER corners
    roi_list = [(x, y, w, h)
                for w in range(int(image_width * MIN_OBJECT_WIDTH_RATIO), int(image_width * MAX_OBJECT_WIDTH_RATIO))
                for h in range(int(image_height * MIN_OBJECT_HEIGHT_RATIO), int(image_height * MAX_OBJECT_HEIGHT_RATIO))
                for x in range(0, image_width - w, image_width * SCANNING_WINDOW_STEP_RATIO_WIDTH)
                for y in range(0, image_height - h, image_height * SCANNING_WINDOW_STEP_RATIO_HEIGHT)]
    return roi_list