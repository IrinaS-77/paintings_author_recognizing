from typing import List

import cv2
import numpy as np
from domain.recognizer.base_recognizer import BaseClassifier


class SIFTClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение признаков
    с помощью гистограммы цветов.
    """

    def get_features(self, image: np.ndarray) -> List:
        #convert to grayscale image
        #gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #initialize SIFT object
        #sift = cv2.SIFT_create()

        #detect keypoints
        #_, desc= sift.detectAndCompute(gray_scale, None)

        #desc = cv2.resize(desc, (250, 250))
        #desc = desc.reshape(250, 250)

        filters = []
        ksize = 51
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
        accum = np.zeros_like(image)
        for kern in filters:
            fimg = cv2.filter2D(image, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
        return accum



