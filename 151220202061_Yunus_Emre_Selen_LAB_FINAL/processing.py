import numpy as np
from skimage import color, filters
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.segmentation import chan_vese, morphological_chan_vese

class ImageProcessor:
    """
    @brief Abstract base class for image processing tasks.

    This class defines the common interface for all image processing operations.
    """

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        @brief Processes an image.

        This method is abstract and must be implemented by subclasses to process
        the image according to the desired functionality.

        @param image: The input image to be processed.
        @return: The processed image.
        """
        raise NotImplementedError


# --- Conversion ---
class Rgb2GrayProcessor(ImageProcessor):
    """
    @brief Converts an RGB image to grayscale.

    This processor converts an RGB image into grayscale using standard
    luminance conversion.
    """
    
    def process(self, image):
        """
        @brief Converts an RGB image to grayscale.

        @param image: The input RGB image.
        @return: The grayscale image.
        """
        return (color.rgb2gray(image) * 255).astype(np.uint8)


class Rgb2HsvProcessor(ImageProcessor):
    """
    @brief Converts an RGB image to HSV color space.

    This processor converts an RGB image to the HSV color space.
    """
    
    def process(self, image):
        """
        @brief Converts an RGB image to HSV.

        @param image: The input RGB image.
        @return: The HSV image.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        hsv = color.rgb2hsv(image)
        return (hsv * 255).astype(np.uint8)


# --- Segmentation ---
class MultiOtsuProcessor(ImageProcessor):
    """
    @brief Performs Multi-Otsu thresholding for segmentation.

    This processor segments an image into multiple regions using the Multi-Otsu method.
    """
    
    def process(self, image):
        """
        @brief Segments the image using Multi-Otsu thresholding.

        This method converts the image to grayscale and applies Multi-Otsu thresholding
        to segment the image into 3 regions.

        @param image: The input image.
        @return: The segmented image.
        """
        # Convert to grayscale and calculate thresholds for 3 classes
        gray = color.rgb2gray(image) if image.ndim == 3 else image
        thresholds = threshold_multiotsu(gray, classes=3)
        result = np.digitize(gray, bins=thresholds)
        # Scale to 0–255 range
        return (result * (255 // result.max())).astype(np.uint8)


class ChanVeseProcessor(ImageProcessor):
    """
    @brief Performs Chan-Vese segmentation.

    This processor segments the image using the Chan-Vese method, which is an active contour
    model for image segmentation.
    """
    
    def process(self, image):
        """
        @brief Segments the image using the Chan-Vese method.

        This method applies the Chan-Vese algorithm to segment the image into regions.

        @param image: The input image.
        @return: The segmented image.
        """
        gray = color.rgb2gray(image) if image.ndim == 3 else image
        # Classic Chan-Vese segmentation
        mask = chan_vese(
            gray,
            mu=0.25,
            lambda1=1,
            lambda2=1,
            tol=1e-3,
            max_num_iter=200,
            extended_output=False
        )
        return (mask.astype(np.uint8) * 255)


class MorphSnakesProcessor(ImageProcessor):
    """
    @brief Performs morphological snakes segmentation.

    This processor segments the image using morphological active contours (Morphological Snakes).
    """
    
    def process(self, image):
        """
        @brief Segments the image using morphological snakes.

        This method applies morphological snakes to segment the image.

        @param image: The input image.
        @return: The segmented image.
        """
        gray = color.rgb2gray(image) if image.ndim == 3 else image
        # Morphological snakes (morphological_chan_vese)
        mask = morphological_chan_vese(gray, num_iter=50)
        return (mask.astype(np.uint8) * 255)


# --- Edge Detection with emphasize ---
def _edge_emphasize(res):
    """
    @brief Emphasizes edges in the image.

    This function performs edge enhancement by normalizing the result and applying
    Otsu's thresholding.

    @param res: The image or edge response to be enhanced.
    @return: The enhanced edge image.
    """
    res = res - res.min()
    if res.max() > 0:
        res /= res.max()
    thr = threshold_otsu(res)
    out = (res - thr) / (1 - thr) if (1 - thr) > 0 else res
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


class RobertsProcessor(ImageProcessor):
    """
    @brief Performs Roberts edge detection.

    This processor applies the Roberts edge detection filter to an image.
    """
    
    def process(self, image):
        """
        @brief Applies Roberts edge detection.

        @param image: The input image.
        @return: The edge-detected image using the Roberts filter.
        """
        gray = color.rgb2gray(image) if image.ndim == 3 else image
        res = filters.roberts(gray)
        return _edge_emphasize(res)


class SobelProcessor(ImageProcessor):
    """
    @brief Performs Sobel edge detection.

    This processor applies the Sobel edge detection filter to an image.
    """
    
    def process(self, image):
        """
        @brief Applies Sobel edge detection.

        @param image: The input image.
        @return: The edge-detected image using the Sobel filter.
        """
        gray = color.rgb2gray(image) if image.ndim == 3 else image
        res = filters.sobel(gray)
        return _edge_emphasize(res)


class ScharrProcessor(ImageProcessor):
    """
    @brief Performs Scharr edge detection.

    This processor applies the Scharr edge detection filter to an image.
    """
    
    def process(self, image):
        """
        @brief Applies Scharr edge detection.

        @param image: The input image.
        @return: The edge-detected image using the Scharr filter.
        """
        gray = color.rgb2gray(image) if image.ndim == 3 else image
        res = filters.scharr(gray)
        return _edge_emphasize(res)


class PrewittProcessor(ImageProcessor):
    """
    @brief Performs Prewitt edge detection.

    This processor applies the Prewitt edge detection filter to an image.
    """
    
    def process(self, image):
        """
        @brief Applies Prewitt edge detection.

        @param image: The input image.
        @return: The edge-detected image using the Prewitt filter.
        """
        gray = color.rgb2gray(image) if image.ndim == 3 else image
        res = filters.prewitt(gray)
        return _edge_emphasize(res)
