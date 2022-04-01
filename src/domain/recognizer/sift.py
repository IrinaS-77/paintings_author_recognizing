from typing import List
from domain.recognizer.base_recognizer import BaseClassifier
import numpy as np
import cv2



class ORBClassifier(BaseClassifier):
    """
    Классификатор, реализующий извлечение
    признаков методом ORB.
    """

    def get_features(self, image: np.ndarray) -> List:
        """Извлечение признаков методом ORB.

        Args:
            image:
                изображение.

        Returns:
            List:
                вектор признаков.
        """
        #orb = cv2.ORB_create()
        # find the keypoints with ORB
        #kp = orb.detect(image,None)
        # compute the descriptors with ORB
        #kp, des = orb.compute(image, kp)

        #des = cv2.resize(des, (250, 250))
        #des = des.reshape(250, 250)

        #return des

        # convert to grayscale image
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # initialize SIFT object
        sift = cv2.SIFT_create()

        # detect keypoints
        _, desc= sift.detectAndCompute(gray_scale, None)

        desc = cv2.resize(desc, (250, 250))
        desc = desc.reshape(250, 250)

        return desc

