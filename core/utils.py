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
