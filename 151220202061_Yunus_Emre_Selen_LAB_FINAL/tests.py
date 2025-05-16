import unittest
import numpy as np
from processing import Rgb2GrayProcessor, MultiOtsuProcessor
from commands import CommandHistory, GrayscaleCommand, ClearSourceCommand
from utils import open_image

class TestProcessing(unittest.TestCase):
    """
    @brief Tests for image processing functionalities.
    """
    def setUp(self):
        """
        @brief Initializes test case with a 3x3 image.
        """
        self.rgb = np.ones((3,3,3), dtype=np.uint8) * 128

    def test_rgb2gray(self):
        """
        @brief Tests conversion from RGB to grayscale.
        """
        gray = Rgb2GrayProcessor().process(self.rgb)
        self.assertEqual(gray.shape, (3,3))
        self.assertTrue((gray == 128).all())

    def test_multiotsu(self):
        """
        @brief Tests multi-Otsu segmentation.
        """
        gray = Rgb2GrayProcessor().process(self.rgb)
        seg  = MultiOtsuProcessor().process(gray)
        self.assertTrue((seg == seg[0,0]).all())
