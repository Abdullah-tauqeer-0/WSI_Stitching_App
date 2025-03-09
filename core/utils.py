import os
import re
import cv2
import numpy as np
from typing import Optional, Tuple

# Global variable to store VIPS bin path if manually set
VIPS_BIN_PATH = None

def init_vips(custom_path: Optional[str] = None) -> bool:
    """
    Initializes libvips.
    If custom_path is provided, it attempts to load from there.
    Otherwise, it checks if vips is already in PATH.
    Returns True if successful, False otherwise.
    """
    global VIPS_BIN_PATH
    
    if custom_path:
        if os.path.exists(custom_path):
            os.environ['PATH'] = os.pathsep.join((custom_path, os.environ['PATH']))
            VIPS_BIN_PATH = custom_path
            
            # Try importing to verify
            try:
                from cffi import FFI
                ffi = FFI()
                # Attempt to load a common DLL to check
                # Note: The specific DLL name might vary by version, so we might skip explicit dlopen
                # and rely on pyvips import.
                import pyvips
                return True
            except Exception as e:
                print(f"Failed to load VIPS from {custom_path}: {e}")
                return False
        else:
            return False

    # Try default import
    try:
        import pyvips
        return True
    except ImportError:
        return False
    except Exception:
        return False

def parse_row_col(filename: str, columns: int) -> Optional[Tuple[int, int]]:
    """Parses row and column from filename."""
    m = re.search(r"_R(\d+)_C(\d+)", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r".*?_([0-9]+)_\d+\.tif$", filename)
    if m:
        num = int(m.group(1))
        return num // columns, num % columns
    return None

def load_image(path: str) -> Optional[np.ndarray]:
    """Loads an image using OpenCV."""
    return cv2.imread(path, cv2.IMREAD_COLOR)

# Reviewed by AT on 2025-03-08