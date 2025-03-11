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
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"row_{row_number}.tif")
    if os.path.exists(cache_path):
        return cv2.imread(cache_path, cv2.IMREAD_COLOR)

    dir_path_abs = os.path.abspath(dir_path)
    tile_infos = []
    for f in os.listdir(dir_path_abs):
        if f.lower().endswith(".tif"):
            rc = parse_row_col(f, columns)
            if rc and rc[0] == row_number:
                tile_infos.append((rc[1], os.path.join(dir_path_abs, f)))
    
    if not tile_infos:
        return None

    tile_infos.sort(key=lambda x: x[0])
    images = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, path): col for col, path in tile_infos}
        for future in concurrent.futures.as_completed(futures):
            col = futures[future]
            img = future.result()
            if img is not None:
                images.append((col, img))
    images.sort(key=lambda x: x[0])
    if not images:
        return None

    positions = [(0, 0)]
    cumulative_x = 0
    cumulative_y = 0

    for i in range(1, len(images)):
        prev_img = images[i-1][1]
        curr_img = images[i][1]
        crop_width_prev = min(1200, prev_img.shape[1])
        crop_width_curr = min(1200, curr_img.shape[1])
        left_crop = prev_img[:, -crop_width_prev:]
        right_crop = curr_img[:, :crop_width_curr]

        try:
            dx, dy, _ = ImageRegistration.get_img2_offset(left_crop, right_crop)
            if dx < 0: dx = 0
            cumulative_x += (prev_img.shape[1] - crop_width_prev) + dx
            cumulative_y += dy
        except Exception:
            fallback_dx = int(0.6 * prev_img.shape[1])
            cumulative_x += fallback_dx
        
        positions.append((cumulative_x, cumulative_y))

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    widths = [img.shape[1] for _, img in images]
    heights = [img.shape[0] for _, img in images]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(x + w for (x, _), w in zip(positions, widths))
    max_y = max(y + h for (y, _), h in zip(positions, heights))
    canvas_width = max_x - min_x
    canvas_height = max_y - min_y
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    tile_centers_y = []
    for (col, img), (x, y) in zip(images, positions):
        h, w = img.shape[:2]
        x_offset = x - min_x
        y_offset = y - min_y
        roi = canvas[y_offset:y_offset+h, x_offset:x_offset+w]
        mask = (roi.sum(axis=2) == 0)
        roi[mask] = img[mask]
        non_mask = ~mask
        if non_mask.any():
            roi[non_mask] = ((roi[non_mask].astype(np.float32) + img[non_mask].astype(np.float32)) / 2).astype(np.uint8)
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = roi
        tile_centers_y.append(y_offset + h // 2)
    
    median_y = int(np.median(tile_centers_y))
    tol = (images[0][1].shape[0]//2)
    crop_top = max(0, median_y - tol)
    crop_bottom = min(canvas.shape[0], median_y + tol)
    cropped_canvas = canvas[crop_top:crop_bottom, :]
    
    cv2.imwrite(cache_path, cropped_canvas)
    return cropped_canvas

def stitch_rows_iteratively(row_images: List[Tuple[int, np.ndarray]], overlap: float) -> Optional[np.ndarray]:
    row_images.sort(key=lambda x: x[0])
    toi = 0.2

    def compute_offset(i: int, top_img: np.ndarray, bottom_img: np.ndarray):
        comp_h = top_img.shape[0]
        cand_h = bottom_img.shape[0]
        comp_overlap = top_img[int(overlap * comp_h - toi * comp_h):, :]
        cand_overlap = bottom_img[:int(overlap * cand_h + toi * cand_h), :]
        try:
            dx, dy, _ = ImageRegistration.get_img2_offset(comp_overlap, cand_overlap)
            flag_ = (cand_overlap.shape[0] * 0.2 < abs(dy)) and (abs(dy) < cand_overlap.shape[0] * 0.5)
            if not flag_:
                dx = 0
                dy = int(overlap * comp_h)
            flag = 0
        except Exception:
            dx = 0
            dy = int(overlap * comp_h)
            flag = 1
        return (i, dx, dy, flag, comp_h, comp_overlap.shape[0])
    
