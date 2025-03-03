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
