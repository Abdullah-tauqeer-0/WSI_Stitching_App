import os
import cv2
import traceback
import concurrent.futures
from PyQt5.QtCore import QThread, pyqtSignal
from .stitching import concatenate_row_sift_return, stitch_rows_iteratively
from .utils import parse_row_col

class StitchingWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, input_dir: str, output_dir: str, columns: int,
                 overlap_y: float, overlap_x: float, compression: str):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.columns = columns
        self.overlap_y = overlap_y
        self.overlap_x = overlap_x
        self.compression = compression

    def run(self):
        try:
            # Late import to ensure VIPS is initialized
