import os
import re
import cv2
import numpy as np
from typing import Optional, Tuple

# Global variable to store VIPS bin path if manually set
VIPS_BIN_PATH = None

def init_vips(custom_path: Optional[str] = None) -> bool:
