# WSI Stitching App

A high-performance, GUI-based application for stitching Whole Slide Images (WSI) from individual tile images.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Features

-   **Hybrid Registration**: Combines **SIFT** (Scale-Invariant Feature Transform) and **Phase Correlation** for robust image alignment.
-   **High-Resolution Output**: Generates **Pyramidal TIFFs** (BigTIFF) compatible with digital pathology viewers (e.g., QuPath, Aperio ImageScope).
-   **Performance**: Utilizes **multi-threading** for parallel image loading and processing.
-   **Modern UI**: Built with **PyQt5**, featuring a clean, dark-themed interface with real-time progress tracking.
-   **VIPS Integration**: Leverages `libvips` for efficient memory management and large image saving.

## 🛠️ Installation

### Prerequisites
-   Python 3.10 or higher
-   [libvips](https://github.com/libvips/libvips/releases) (v8.10+)

### Dependencies
Install the required Python packages:

```bash
pip install opencv-python-headless numpy PyQt5 pyvips
```

## 💻 Usage

1.  **Launch the Application**:
    Double-click `run_app.bat` or run:
    ```bash
    python -m WSI_App.main
    ```

2.  **Configure**:
    -   **Input Directory**: Select the folder containing your `.tif` tiles.
    -   **Output Directory**: Select where to save the stitched WSI.
    -   **Parameters**: Adjust overlap (default 40-48%) and columns if necessary.

3.  **Run**:
    Click **Start Stitching**. The log window will show detailed progress.

## ⚙️ Configuration

If `libvips` is not in your system PATH, you can specify its location in the **Advanced** section of the UI.

## 🏗️ Architecture

The application is modularized for maintainability:
-   `ui.py`: Handles the Graphical User Interface.
-   `worker.py`: Manages background threads to keep the UI responsive.
-   `stitching.py`: Contains the core computer vision algorithms.
-   `utils.py`: Helper functions and VIPS initialization.

## 📝 License

This project is licensed under the MIT License.
