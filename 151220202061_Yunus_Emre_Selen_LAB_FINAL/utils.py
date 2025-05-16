# utils.py
from PyQt5.QtWidgets import QMessageBox
from skimage import io
import numpy as np

def open_image(path: str) -> np.ndarray:
    """
    @brief Opens an image from the specified file path.

    This function reads an image from the given file path and converts it to a NumPy array.
    If the image is in float format, it will be scaled to the [0, 255] range and cast to uint8.
    
    @param path: The file path of the image to be opened.
    @return: A NumPy array representing the image.
    @throws IOError: If the image cannot be opened from the specified path.
    """
    img = io.imread(path)
    if img is None:
        raise IOError(f"Cannot open image: {path}")
    # Convert to uint8 if the image is in float [0–1] format
    if img.dtype == np.float64 or img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)
    # Keep the 2D array if the image is grayscale, else return [H, W, 3] uint8 for color images
    return img

def save_image(img: np.ndarray, path: str):
    """
    @brief Saves the given image to the specified file path.

    This function attempts to save the NumPy array image to the given path. If an error occurs, it raises
    an IOError with the specific error message.

    @param img: The image to be saved, represented as a NumPy array.
    @param path: The destination file path where the image will be saved.
    @throws IOError: If the image cannot be saved to the specified path.
    """
    try:
        io.imsave(path, img)
    except Exception as e:
        raise IOError(f"Failed to save image: {path}\n{e}")

def export_image(img: np.ndarray, path: str):
    """
    @brief Exports the image to the specified file path.

    This function is a wrapper around `save_image` to provide a consistent interface for exporting images.
    
    @param img: The image to be exported, represented as a NumPy array.
    @param path: The file path where the image will be exported.
    """
    save_image(img, path)

def show_error(msg: str):
    """
    @brief Displays an error message in a popup window.

    This function creates a message box with a critical error icon and the provided message text.

    @param msg: The error message to be displayed in the message box.
    """
    dlg = QMessageBox()
    dlg.setIcon(QMessageBox.Critical)
    dlg.setWindowTitle("Error")
    dlg.setText(msg)
    dlg.exec_()
