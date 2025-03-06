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

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        diff = (src_pts - dst_pts).reshape(-1, 2)
        num_matches = len(good_matches)
        
        # Simple outlier removal
        para = 2
        if len(diff) > 2 * para:
            rx = np.concatenate((np.argsort(diff[:, 0])[:para], np.argsort(diff[:, 0])[-para:]))
            ry = np.concatenate((np.argsort(diff[:, 1])[:para], np.argsort(diff[:, 1])[-para:]))
            remove_idx = np.unique(np.concatenate((rx, ry)))
            keep_idx = np.array([i for i in range(len(diff)) if i not in remove_idx])
            src_pts = src_pts[keep_idx]
            dst_pts = dst_pts[keep_idx]

        dif = np.mean(src_pts - dst_pts, axis=(0, 1))
        diff_val = np.array([int(round(dif[0])), int(round(dif[1]))])

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            dx, dy = ImageRegistration.phase_correlation_offset(gray1, gray2)
            return dx, dy, len(good_matches)

        t_x = int(round(H[0, 2]))
        t_y = int(round(H[1, 2]))
        offset_val = np.array([t_x, t_y])

        if np.linalg.norm(diff_val - offset_val) > 2:
            return diff_val[0], diff_val[1], num_matches
        else:
            return offset_val[0], offset_val[1], num_matches

    @staticmethod
    def phase_correlation_offset(gray1: np.ndarray, gray2: np.ndarray) -> Tuple[int, int]:
        f1 = np.float32(gray1)
        f2 = np.float32(gray2)

        if f1.shape[1] != f2.shape[1]:
            new_width = min(f1.shape[1], f2.shape[1])
            f1 = f1[:, (f1.shape[1] - new_width) // 2:(f1.shape[1] + new_width) // 2]
            f2 = f2[:, (f2.shape[1] - new_width) // 2:(f2.shape[1] + new_width) // 2]

        if f1.shape[0] != f2.shape[0]:
            new_height = min(f1.shape[0], f2.shape[0])
            f1 = f1[(f1.shape[0] - new_height) // 2:(f1.shape[0] + new_height) // 2, :]
            f2 = f2[(f2.shape[0] - new_height) // 2:(f2.shape[0] + new_height) // 2, :]

        def compute_phase_correlate(a, b):
            return cv2.phaseCorrelate(a, b)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(compute_phase_correlate, f1, f2)
                shift, conf = future.result(timeout=20)
                if conf < 0.5:
                    return 0, 0
        except Exception:
            return 0, 0

        dx = int(round(shift[0]))
        dy = int(round(shift[1]))
        return dx, dy

def concatenate_row_sift_return(row_number: int, dir_path: str, columns: int, overlap: float, cache_dir: str = "cached_rows") -> Optional[np.ndarray]:
