import cv2 as cv
import numpy as np


class ImageProcessor:
    """Class for image processing.

    Attributes:

    """

    def __init__(self, fp=None):
        """Initialize image process class.

        Args:
            fp (str) : File path to the image.
        """
        if fp is not None:
            self.load_img(fp)
        else:
            self.img = None
            self.width = None
            self.height = None
            self.channels = None

    def load_img(self, fp):
        self.img = cv.imread(fp)
        cv.cvtColor(self.img, cv.COLOR_BGR2RGB, self.img)

        # Get height, width, and channels from image.
        self.height, self.width, self.channels = self.img.shape

    def get_img(self):
        """Get image"""
        return self.img

    def save_img(self, fp):
        cv.cvtColor(self.img, cv.COLOR_BGR2RGB, self.img)
        cv.imwrite(fp, self.img)

    # def show(self, img=None, name='Image'):
    #     """Display image. Press 'esc' to exit.
    #
    #     Args:
    #         img (numpy.array): Image array representation.
    #         name (str): Name of the window.
    #     """
    #     if img is None:
    #         img = self.img
    #
    #     cv.imshow(name, img)
    #     if cv.waitKey(0) == 27:
    #         cv.destroyAllWindows()

    def get_split_color(self, mode):
        """Split image color

        Args:
            mode (str): b - blue; r - red; g - green.
        """
        if mode == 'b':
            name = 'Blue'
            img = self.img[:, :, 2]
        elif mode == 'r':
            name = 'Red'
            img = self.img[:, :, 0]
        elif mode == 'g':
            name = ' Green'
            img = self.img[:, :, 1]
        else:
            raise Exception("Color option not exist!")

        self.show(img, name)

    def get_pixel_color(self, height, width):
        """Get pixel color

        Args:
            height (int): Height position.
            width (int): Width position.

        Returns:
            A tuple of rgb color.
        """
        return self.img[height][width]

    def get_shape(self):
        """Get image shape.

        Returns:
            Height, width and number of channels.
        """
        return self.height, self.width, self.channels
