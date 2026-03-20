import cv2
import os
import numpy as np


class ColorSegmenter:
    """
    Isolates walls using RGB color thresholds
    to generate a high-resolution (30x30) navigable binary matrix
    Includes a built-in GUI tuner for calibrating lighting setups.
    """

    def __init__(self, img_size=640, grid_size=30):
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size / grid_size

        # Default RGB values for purely white walls.
        # Format: [Red, Green, Blue] (0-255 scale)
        self.lower_wall_rgb = np.array([200, 200, 200])
        self.upper_wall_rgb = np.array([255, 255, 255])
