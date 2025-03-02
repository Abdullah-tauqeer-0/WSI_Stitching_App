import os
import cv2
import numpy as np
import concurrent.futures
from typing import Tuple, List, Optional
from .utils import load_image, parse_row_col

class ImageRegistration:
    @staticmethod
    def get_img2_offset(img1: np.ndarray, img2: np.ndarray) -> Tuple[int, int, int]:
        """
        Compute the translation (dx, dy) between two images.
        """
        gray1 = img1
        gray2 = img2
        
        # Ensure grayscale
        if gray1.ndim == 3: gray1 = cv2.cvtColor(gray1, cv2.COLOR_BGR2GRAY)
        if gray2.ndim == 3: gray2 = cv2.cvtColor(gray2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            dx, dy = ImageRegistration.phase_correlation_offset(gray1, gray2)
            return dx, dy, 0

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]

        if len(good_matches) < 12:
            dx, dy = ImageRegistration.phase_correlation_offset(gray1, gray2)
            return dx, dy, len(good_matches)
