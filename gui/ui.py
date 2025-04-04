import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, 
    QProgressBar, QComboBox, QGroupBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt
from core.worker import StitchingWorker
from core.utils import init_vips

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WSI Stitching App")
        self.resize(800, 600)
        self.worker = None
        
        # Apply Dark Theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }
            QLineEdit { background-color: #3b3b3b; border: 1px solid #555; border-radius: 3px; padding: 5px; color: #fff; }
            QPushButton { background-color: #0d6efd; color: white; border: none; border-radius: 4px; padding: 8px 16px; font-weight: bold; }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:disabled { background-color: #555; }
            QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #198754; width: 10px; }
            QTextEdit { background-color: #1e1e1e; border: 1px solid #555; font-family: Consolas, monospace; font-size: 12px; }
            QLabel { color: #ccc; }
        """)
        
