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
        
        self.setup_ui()
        self.check_vips()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Header
        header = QLabel("WSI Stitching Application")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Configuration Group
        config_group = QGroupBox("Configuration")
        config_layout = QFormLayout()
        
        # Input
        input_layout = QHBoxLayout()
        self.input_line_edit = QLineEdit()
        self.input_line_edit.setPlaceholderText("Select directory containing .tif tiles")
        self.input_browse_btn = QPushButton("Browse")
        self.input_browse_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_line_edit)
        input_layout.addWidget(self.input_browse_btn)
        config_layout.addRow("Input Directory:", input_layout)

        # Output
        output_layout = QHBoxLayout()
        self.output_line_edit = QLineEdit()
        self.output_line_edit.setPlaceholderText("Select output directory")
        self.output_browse_btn = QPushButton("Browse")
        self.output_browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_line_edit)
        output_layout.addWidget(self.output_browse_btn)
        config_layout.addRow("Output Directory:", output_layout)

        # Parameters
        self.columns_line_edit = QLineEdit("61")
        config_layout.addRow("Columns:", self.columns_line_edit)

        self.overlap_y_line_edit = QLineEdit("0.48")
        config_layout.addRow("Vertical Overlap (0-1):", self.overlap_y_line_edit)

        self.overlap_x_line_edit = QLineEdit("0.4")
        config_layout.addRow("Horizontal Overlap (0-1):", self.overlap_x_line_edit)

        self.compression_combo = QComboBox()
        self.compression_combo.addItems(["jpeg", "lzw", "deflate", "none"])
        self.compression_combo.setStyleSheet("background-color: #3b3b3b; color: white; padding: 5px;")
        config_layout.addRow("Compression:", self.compression_combo)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # VIPS Configuration (Hidden by default unless needed, but good to have access)
        vips_group = QGroupBox("Advanced")
        vips_layout = QHBoxLayout()
        self.vips_path_edit = QLineEdit()
        self.vips_path_edit.setPlaceholderText("Path to VIPS bin folder (optional)")
        self.vips_browse_btn = QPushButton("Locate VIPS")
        self.vips_browse_btn.clicked.connect(self.browse_vips)
        vips_layout.addWidget(QLabel("VIPS Path:"))
        vips_layout.addWidget(self.vips_path_edit)
        vips_layout.addWidget(self.vips_browse_btn)
        vips_group.setLayout(vips_layout)
        main_layout.addWidget(vips_group)

        # Status & Progress
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: #0dcaf0;")
        main_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Log
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        main_layout.addWidget(self.log_text_edit)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Start Stitching")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #198754; font-size: 16px;")
        self.run_btn.clicked.connect(self.run_stitching)
        btn_layout.addWidget(self.run_btn)
        main_layout.addLayout(btn_layout)

    def check_vips(self):
        # Initial check
        if not init_vips():
            self.append_log("Warning: VIPS not found in PATH. Please locate the 'bin' folder in the Advanced section.")
            self.status_label.setText("VIPS Missing")
        else:
            self.append_log("VIPS initialized successfully.")

    def browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path: self.input_line_edit.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path: self.output_line_edit.setText(path)

    def browse_vips(self):
        path = QFileDialog.getExistingDirectory(self, "Select VIPS bin Directory")
        if path:
            self.vips_path_edit.setText(path)
            if init_vips(path):
                self.append_log(f"VIPS initialized from {path}")
                self.status_label.setText("Ready")
            else:
                self.append_log(f"Failed to initialize VIPS from {path}")

    def append_log(self, message: str):
        self.log_text_edit.append(message)

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def update_status(self, text: str):
        self.status_label.setText(text)

    def run_stitching(self):
        input_dir = self.input_line_edit.text().strip()
        output_dir = self.output_line_edit.text().strip()
        
        # Ensure VIPS is ready
        vips_path = self.vips_path_edit.text().strip()
        if vips_path:
            init_vips(vips_path)

        if not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Error", "Invalid input directory.")
            return
        if not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Error", "Invalid output directory.")
            return

        try:
            columns = int(self.columns_line_edit.text().strip())
            overlap_y = float(self.overlap_y_line_edit.text().strip())
            overlap_x = float(self.overlap_x_line_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid numeric parameters.")
            return

        compression = self.compression_combo.currentText()

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        self.append_log("Starting stitching process...")
        
        self.worker = StitchingWorker(input_dir, output_dir, columns, overlap_y, overlap_x, compression)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.status_signal.connect(self.update_status)
        self.worker.finished_signal.connect(self.stitching_finished)
        self.worker.start()

    def stitching_finished(self, output_path: str):
        if output_path:
            self.append_log(f"Process completed. Output saved to: {output_path}")
            QMessageBox.information(self, "Success", f"Stitching Complete!\nSaved to: {output_path}")
        else:
            self.append_log("Stitching process finished with errors.")
            QMessageBox.critical(self, "Error", "Stitching failed. Check log for details.")
        
        self.run_btn.setEnabled(True)
        self.status_label.setText("Idle")
        self.progress_bar.setValue(0)

# Reviewed by AT on 2025-05-11
# Refactor pending for v2
# TODO: Optimize this section 18
# Fixed edge case 971
# Reviewed by AT on 2025-05-24
# Fixed edge case 344
# TODO: Optimize this section 86
# Reviewed by AT on 2025-06-16
# TODO: Optimize this section 78