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
            import pyvips
            
            self.status_signal.emit("Starting stitching process...")
            overlap_y = self.overlap_y
            overlap_x_norm = 1 - self.overlap_x
            dir_path = self.input_dir
            columns = self.columns

            # Compute available row numbers
            dir_path_abs = os.path.abspath(dir_path)
            row_set = set()
            for f in os.listdir(dir_path_abs):
                if f.lower().endswith(".tif"):
                    rc = parse_row_col(f, columns)
                    if rc:
                        row_set.add(rc[0])
            
            if not row_set:
                self.log_signal.emit("No valid rows detected in the directory.")
                self.finished_signal.emit("")
                return
            
            row_numbers = sorted(list(row_set))
            self.log_signal.emit(f"Detected rows: {row_numbers}")

