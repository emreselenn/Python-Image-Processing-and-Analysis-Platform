from abc import ABC, abstractmethod
from processing import (
    Rgb2GrayProcessor, Rgb2HsvProcessor,
    MultiOtsuProcessor, ChanVeseProcessor, MorphSnakesProcessor,
    RobertsProcessor, SobelProcessor, ScharrProcessor, PrewittProcessor
)

class ICommand(ABC):
    """
    @brief Abstract base class for command pattern implementation.

    This class defines the interface for all concrete commands to implement 
    the execute and undo operations.
    """
    
    @abstractmethod
    def execute(self):
        """
        @brief Execute the command.
        
        This method is implemented by subclasses to execute the command.
        """
        pass

    @abstractmethod
    def undo(self):
        """
        @brief Undo the command.

        This method is implemented by subclasses to undo the command.
        """
        pass

class CommandHistory:
    """
    @brief Maintains history of commands for undo and redo operations.

    This class stores the stack of executed commands and supports undo/redo operations.
    """
    
    def __init__(self):
        """
        @brief Initializes the command history with empty stacks.
        """
        self._undo_stack = []
        self._redo_stack = []

    def push(self, cmd: ICommand):
        """
        @brief Pushes a command to the undo stack and clears the redo stack.

        @param cmd: The command to be pushed to history.
        """
        self._undo_stack.append(cmd)
        self._redo_stack.clear()

    def undo(self):
        """
        @brief Undoes the last executed command, if available.
        """
        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)

    def redo(self):
        """
        @brief Redoes the last undone command, if available.
        """
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)

# --- Concrete Commands ---
class BaseImageCommand(ICommand):
    """
    @brief Base class for image processing commands.

    This command operates on images and supports undo functionality by storing 
    the previous image state.
    """
    
    def __init__(self, ctrl):
        """
        @brief Initializes the BaseImageCommand with a controller.

        @param ctrl: The controller that holds the source and output images.
        """
        self.ctrl = ctrl
        self.prev = ctrl.img_output.copy() if ctrl.img_output is not None else None

    def undo(self):
        """
        @brief Restores the previous image state.
        """
        self.ctrl.img_output = self.prev
        self.ctrl._show_image(self.prev, self.ctrl.lblOutputImage)

class ClearSourceCommand(ICommand):
    """
    @brief Command to clear the source image.
    """

    def __init__(self, ctrl):
        """
        @brief Initializes the command to clear the source image.

        @param ctrl: The controller with the image source to be cleared.
        """
        self.ctrl = ctrl
        self.prev_src = ctrl.img_source

    def execute(self):
        """
        @brief Clears the source image.
        """
        self.ctrl.img_source = None
        self.ctrl.lblSourceImage.clear()

    def undo(self):
        """
        @brief Restores the previous source image.
        """
        self.ctrl.img_source = self.prev_src
        self.ctrl._show_image(self.prev_src, self.ctrl.lblSourceImage)

class ClearOutputCommand(ICommand):
    """
    @brief Command to clear the output image.
    """
    
    def __init__(self, ctrl):
        """
        @brief Initializes the command to clear the output image.

        @param ctrl: The controller with the output image to be cleared.
        """
        self.ctrl = ctrl

    def execute(self):
        """
        @brief Clears the output image.
        """
        self.ctrl.img_output = None
        self.ctrl.lblOutputImage.clear()

    def undo(self):
        """
        @brief No operation for undo since output can’t be restored.
        """
        pass

# Conversion commands
class GrayscaleCommand(BaseImageCommand):
    """
    @brief Command to convert the source image to grayscale.
    """
    
    def execute(self):
        """
        @brief Applies grayscale conversion on the source image.
        """
        proc = Rgb2GrayProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)

class HsvCommand(BaseImageCommand):
    """
    @brief Command to convert the source image to HSV.
    """
    
    def execute(self):
        """
        @brief Applies HSV conversion on the source image.
        """
        proc = Rgb2HsvProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)

# Segmentation commands
class MultiOtsuCommand(BaseImageCommand):
    """
    @brief Command to apply multi-Otsu segmentation on the source image.
    """
    
    def execute(self):
        """
        @brief Applies multi-Otsu thresholding segmentation on the source image.
        """
        proc = MultiOtsuProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)

class ChanVeseCommand(BaseImageCommand):
    """
    @brief Command to apply Chan-Vese segmentation on the source image.
    """
    
    def execute(self):
        """
        @brief Applies Chan-Vese segmentation on the source image.
        """
        proc = ChanVeseProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)

class MorphSnakesCommand(BaseImageCommand):
    """
    @brief Command to apply morphological snakes segmentation on the source image.
    """
    
    def execute(self):
        """
        @brief Applies morphological snakes segmentation on the source image.
        """
        proc = MorphSnakesProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)

# Edge Detection commands
class RobertsCommand(BaseImageCommand):
    """
    @brief Command to apply Roberts edge detection on the source image.
    """
    
    def execute(self):
        """
        @brief Applies Roberts edge detection on the source image.
        """
        proc = RobertsProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)

class SobelCommand(BaseImageCommand):
    """
    @brief Command to apply Sobel edge detection on the source image.
    """
    
    def execute(self):
        """
        @brief Applies Sobel edge detection on the source image.
        """
        proc = SobelProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)

class ScharrCommand(BaseImageCommand):
    """
    @brief Command to apply Scharr edge detection on the source image.
    """
    
    def execute(self):
        """
        @brief Applies Scharr edge detection on the source image.
        """
        proc = ScharrProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)

class PrewittCommand(BaseImageCommand):
    """
    @brief Command to apply Prewitt edge detection on the source image.
    """
    
    def execute(self):
        """
        @brief Applies Prewitt edge detection on the source image.
        """
        proc = PrewittProcessor()
        out = proc.process(self.ctrl.img_source)
        self.ctrl.img_output = out
        self.ctrl._show_image(out, self.ctrl.lblOutputImage)
