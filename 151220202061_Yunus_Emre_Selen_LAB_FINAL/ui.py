import os
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from qt_design import Ui_MainWindow
from utils import open_image, save_image, export_image, show_error
from commands import (
    CommandHistory,
    ClearSourceCommand,
    ClearOutputCommand,
    GrayscaleCommand,
    HsvCommand,
    MultiOtsuCommand,
    ChanVeseCommand,
    MorphSnakesCommand,
    RobertsCommand,
    SobelCommand,
    ScharrCommand,
    PrewittCommand
)

class MainWindowController(QMainWindow, Ui_MainWindow):
    """
    @brief Main window controller for the image processing application.

    This class is responsible for managing the main window, handling user interactions,
    and coordinating various image processing tasks such as loading, saving, and applying filters.
    """
    
    def __init__(self):
        """
        @brief Initializes the main window and sets up the initial state.

        Sets up the user interface, initializes the history stack, and prepares
        the image display panels.
        """
        super(MainWindowController, self).__init__()
        self.setupUi(self)
        self.img_source = None
        self.img_output = None
        self.source_path = None
        self.history = CommandHistory()

        # Set scalable image panels
        self.lblSourceImage.setScaledContents(True)
        self.lblOutputImage.setScaledContents(True)
        self.lblSourceImage.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lblOutputImage.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self._connect_signals()
        self._set_initial_state()

    def resizeEvent(self, event):
        """
        @brief Handles window resizing events.

        @param event: The resize event.
        """
        super(MainWindowController, self).resizeEvent(event)
        if self.img_source is not None:
            self._show_image(self.img_source, self.lblSourceImage)
        if self.img_output is not None:
            self._show_image(self.img_output, self.lblOutputImage)

    def _set_initial_state(self):
        """
        @brief Sets the initial state of the application, enabling and disabling controls as needed.

        Initially, only the "Open Source" button is enabled. Other controls are disabled until
        a source image is loaded.
        """
        for ctrl in (
            self.btnOpenSource, self.actionOpenSource,
        ):
            ctrl.setEnabled(True)
        self._disable_output_controls()

        # Disable all other controls
        for ctrl in (
            self.btnClearSource, self.actionClearSource,
            self.btnExportSource, self.actionExportSource,
            self.btnRGB2Gray, self.actionRGB2Gray,
            self.btnRGB2HSV, self.actionRGB2HSV,
            self.btnMultiOtsu, self.actionMultiOtsu,
            self.btnChanVese, self.actionChanVese,
            self.btnMorphSnakes, self.actionMorphSnakes,
            self.btnEdgeRoberts, self.actionEdgeRoberts,
            self.btnEdgeSobel, self.actionEdgeSobel,
            self.btnEdgeScharr, self.actionEdgeScharr,
            self.btnEdgePrewitt, self.actionEdgePrewitt
        ):
            ctrl.setEnabled(False)

    def _disable_output_controls(self):
        """
        @brief Disables output controls when there is no output image.
        """
        for ctrl in (
            self.btnSaveOutput, self.actionSaveOutput,
            self.btnSaveAsOutput, self.actionSaveAsOutput,
            self.btnExportOutput, self.actionExportOutput,
            self.btnClearOutput, self.actionClearOutput,
            self.btnUndoOutput, self.actionUndoOutput,
            self.btnRedoOutput, self.actionRedoOutput
        ):
            ctrl.setEnabled(False)

    def _connect_signals(self):
        """
        @brief Connects signals and slots for user interface actions.

        This function sets up all button click handlers and menu actions for the application.
        """
        # File menu & toolbar
        self.actionOpenSource.triggered.connect(self.open_source)
        self.btnOpenSource.clicked.connect(self.open_source)
        self.actionSaveOutput.triggered.connect(self.save_output)
        self.btnSaveOutput.clicked.connect(self.save_output)
        self.actionSaveAsOutput.triggered.connect(self.save_as_output)
        self.btnSaveAsOutput.clicked.connect(self.save_as_output)
        self.actionExportSource.triggered.connect(self.export_source)
        self.btnExportSource.clicked.connect(self.export_source)
        self.actionExportOutput.triggered.connect(self.export_output)
        self.btnExportOutput.clicked.connect(self.export_output)
        self.actionExit.triggered.connect(self.close)

        # Edit menu
        self.actionClearSource.triggered.connect(self.clear_source)
        self.btnClearSource.clicked.connect(self.clear_source)
        self.actionClearOutput.triggered.connect(self.clear_output)
        self.btnClearOutput.clicked.connect(self.clear_output)
        self.actionUndoOutput.triggered.connect(self.undo)
        self.btnUndoOutput.clicked.connect(self.undo)
        self.actionRedoOutput.triggered.connect(self.redo)
        self.btnRedoOutput.clicked.connect(self.redo)

        # Conversion / Segmentation / Edge Detection
        self.btnRGB2Gray.clicked.connect(self.apply_grayscale)
        self.actionRGB2Gray.triggered.connect(self.apply_grayscale)
        self.btnRGB2HSV.clicked.connect(self.apply_hsv)
        self.actionRGB2HSV.triggered.connect(self.apply_hsv)
        self.btnMultiOtsu.clicked.connect(self.apply_multiotsu)
        self.actionMultiOtsu.triggered.connect(self.apply_multiotsu)
        self.btnChanVese.clicked.connect(self.apply_chanvese)
        self.actionChanVese.triggered.connect(self.apply_chanvese)
        self.btnMorphSnakes.clicked.connect(self.apply_morphsnakes)
        self.actionMorphSnakes.triggered.connect(self.apply_morphsnakes)
        self.btnEdgeRoberts.clicked.connect(self.apply_roberts)
        self.actionEdgeRoberts.triggered.connect(self.apply_roberts)
        self.btnEdgeSobel.clicked.connect(self.apply_sobel)
        self.actionEdgeSobel.triggered.connect(self.apply_sobel)
        self.btnEdgeScharr.clicked.connect(self.apply_scharr)
        self.actionEdgeScharr.triggered.connect(self.apply_scharr)
        self.btnEdgePrewitt.clicked.connect(self.apply_prewitt)
        self.actionEdgePrewitt.triggered.connect(self.apply_prewitt)

    def open_source(self):
        """
        @brief Opens a source image file and displays it.

        Opens the file dialog, allows the user to select an image, and displays it in the source image panel.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.tif)")
        if not path:
            return
        try:
            self.img_source = open_image(path)
            self.source_path = path
            self._show_image(self.img_source, self.lblSourceImage)

            # Enable source processing controls
            for ctrl in (
                self.btnClearSource, self.actionClearSource,
                self.btnExportSource, self.actionExportSource,
                self.btnRGB2Gray, self.actionRGB2Gray,
                self.btnRGB2HSV, self.actionRGB2HSV,
                self.btnMultiOtsu, self.actionMultiOtsu,
                self.btnChanVese, self.actionChanVese,
                self.btnMorphSnakes, self.actionMorphSnakes,
                self.btnEdgeRoberts, self.actionEdgeRoberts,
                self.btnEdgeSobel, self.actionEdgeSobel,
                self.btnEdgeScharr, self.actionEdgeScharr,
                self.btnEdgePrewitt, self.actionEdgePrewitt
            ):
                ctrl.setEnabled(True)

            # Disable output controls initially
            self._disable_output_controls()
        except Exception as e:
            show_error(str(e))

    def save_output(self):
        """
        @brief Saves the processed output image to a file.

        This function saves the output image to a file, generating a unique name if necessary.
        """
        if not self.source_path or self.img_output is None:
            return
        base, ext = os.path.splitext(self.source_path)
        i = 1
        while True:
            candidate = f"{base}_{i}{ext}"
            if not os.path.exists(candidate):
                out_path = candidate
                break
            i += 1
        save_image(self.img_output, out_path)

    def save_as_output(self):
        """
        @brief Saves the processed output image under a new name.

        @return: None
        """
        if not self.source_path or self.img_output is None:
            return
        _, ext = os.path.splitext(self.source_path)
        filter_str = f"{ext.strip('.').upper()} (*{ext})"
        directory = os.path.dirname(self.source_path)
        default = os.path.join(directory, os.path.basename(self.source_path))
        path, _ = QFileDialog.getSaveFileName(self, "Save As", default, filter_str)
        if path:
            save_image(self.img_output, path)

    def export_source(self):
        """
        @brief Exports the source image to a specified file.

        Allows the user to export the source image in a selected format.
        """
        if self.img_source is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Source As", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if path:
            export_image(self.img_source, path)

    def export_output(self):
        """
        @brief Exports the processed output image to a specified file.

        Allows the user to export the processed output image in a selected format.
        """
        if self.img_output is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Output As", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if path:
            export_image(self.img_output, path)

    def clear_source(self):
        """
        @brief Clears the source image and resets the output image.

        This function clears the source image, resets the output, and clears the command history.
        """
        cmd = ClearSourceCommand(self)
        self.history.push(cmd)
        cmd.execute()
        self.img_output = None
        self.lblOutputImage.clear()
        self.history._undo_stack.clear()
        self.history._redo_stack.clear()
        self._disable_output_controls()
        self._set_initial_state()

    def clear_output(self):
        """
        @brief Clears the processed output image.

        Clears the processed output image and resets related controls.
        """
        cmd = ClearOutputCommand(self)
        self.history.push(cmd)
        cmd.execute()
        self.history._undo_stack.clear()
        self.history._redo_stack.clear()
        self._disable_output_controls()

    def undo(self):
        """
        @brief Undoes the last command in the history.

        Reverts the image processing to the previous state by using the undo functionality.
        """
        self.history.undo()
        if self.img_output is None:
            self._disable_output_controls()
        else:
            self._update_undo_redo_buttons()

    def redo(self):
        """
        @brief Redoes the last undone command.

        Re-applies the last undone command from the history.
        """
        self.history.redo()
        if self.img_output is None or not self.history._redo_stack:
            self.btnRedoOutput.setEnabled(False)
            self.actionRedoOutput.setEnabled(False)
        else:
            self.btnRedoOutput.setEnabled(True)
            self.actionRedoOutput.setEnabled(True)

    def _update_undo_redo_buttons(self):
        """
        @brief Updates the enable/disable state of undo/redo buttons based on the command history.
        """
        can_undo = bool(self.history._undo_stack)
        can_redo = bool(self.history._redo_stack)
        for ctrl in (self.btnUndoOutput, self.actionUndoOutput):
            ctrl.setEnabled(can_undo)
        for ctrl in (self.btnRedoOutput, self.actionRedoOutput):
            ctrl.setEnabled(can_redo)

    def _apply(self, CmdClass):
        """
        @brief Applies a specific image processing command.

        Executes the specified command and updates the output image and relevant controls.

        @param CmdClass: The command class to be applied.
        """
        if self.img_source is None:
            return
        cmd = CmdClass(self)
        self.history.push(cmd)
        cmd.execute()
        self.img_output = cmd.ctrl.img_output if hasattr(cmd, 'ctrl') else None

        # Enable output controls
        for ctrl in (
            self.btnSaveOutput, self.actionSaveOutput,
            self.btnSaveAsOutput, self.actionSaveAsOutput,
            self.btnExportOutput, self.actionExportOutput,
            self.btnClearOutput, self.actionClearOutput,
            self.btnUndoOutput, self.actionUndoOutput
        ):
            ctrl.setEnabled(True)

        # Update redo state
        self._update_undo_redo_buttons()

    # Conversion / Segmentation / Edge Detection method bindings
    apply_grayscale   = lambda self: self._apply(GrayscaleCommand)
    apply_hsv         = lambda self: self._apply(HsvCommand)
    apply_multiotsu   = lambda self: self._apply(MultiOtsuCommand)
    apply_chanvese    = lambda self: self._apply(ChanVeseCommand)
    apply_morphsnakes = lambda self: self._apply(MorphSnakesCommand)
    apply_roberts     = lambda self: self._apply(RobertsCommand)
    apply_sobel       = lambda self: self._apply(SobelCommand)
    apply_scharr      = lambda self: self._apply(ScharrCommand)
    apply_prewitt     = lambda self: self._apply(PrewittCommand)

    def _show_image(self, array, label):
        """
        @brief Displays the image in the specified label.

        This method converts the image to an appropriate format and scales it to fit the label.

        @param array: The image array to be displayed.
        @param label: The QLabel where the image will be shown.
        """
        img = array.astype(np.float64)
        if img.ndim == 2:
            img -= img.min()
            if img.max() > 0:
                img /= img.max()
            img = (img * 255).astype(np.uint8)
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            if img.max() > 1:
                img /= 255.0
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            h, w, ch = img.shape
            qimg = QImage(img.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.size(), QtCore.Qt.KeepAspectRatio))
