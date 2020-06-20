import cv2 as cv
import numpy as np
import time
import math


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
            self.processed_img = None
            self.width = None
            self.height = None
            self.channels = None

    def load_img(self, fp):
        self.img = cv.imread(fp)
        cv.cvtColor(self.img, cv.COLOR_BGR2RGB, self.img)
        self.processed_img = self.img
        self.update_img_property()

    def update_img_property(self):
        self.height, self.width, self.channels = self.processed_img.shape

    def get_img(self):
        """Get image"""
        return self.processed_img

    def set_img(self, img):
        self.img = img

    def resore_changes(self):
        self.processed_img = self.img

    def save_img(self, fp):
        cv.cvtColor(self.processed_img, cv.COLOR_BGR2RGB, self.processed_img)
        cv.imwrite(fp, self.processed_img)

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
            img = self.processed_img[:, :, 2]
        elif mode == 'r':
            name = 'Red'
            img = self.processed_img[:, :, 0]
        elif mode == 'g':
            name = ' Green'
            img = self.processed_img[:, :, 1]
        else:
            raise Exception("Color option not exist!")

        self.processed_img = img

    def get_pixel_color(self, height, width):
        """Get pixel color

        Args:
            height (int): Height position.
            width (int): Width position.

        Returns:
            A tuple of rgb color.
        """
        return self.processed_img[height][width]

    def get_shape(self):
        """Get image shape.

        Returns:
            Height, width and number of channels.
        """
        return self.height, self.width, self.channels

    def shift(self, x, y, cut=False):
        """Shift the image vertically and horizontally.

        If cut the shifted image, the part shifted out will not
        apper and the image size remain the same. If not cut the
        image, the blank area will be filled with black. Size of
        image will increse.

        Args:
            x (int): Number of pixels shift on height
            y (int): Number of pixels shift on width
            cut (bool): Cut the image or not.

        Returns:
            Numpy array representation of shifted image.
        """
        transform_mat = np.array([[1, 0, x],
                                  [0, 1, y],
                                  [0, 0, 1]], dtype=np.int8)
        height, width, channels = self.get_shape()

        if not cut:
            size = [height + abs(x), width + abs(y), channels]
            img = np.zeros(size, dtype=np.uint8)

            # start = time.time()
            for i in range(self.height):
                for j in range(self.width):
                    # Get new position
                    src = np.array([i, j, 1], dtype=np.int32)
                    dst = np.dot(transform_mat, src)

                    if x >= 0 and y >= 0:
                        img[dst[0]][dst[1]] = self.img[i][j]
                    elif y >= 0:
                        img[i][dst[1]] = self.img[i][j]
                    elif x >= 0:
                        img[dst[0]][j] = self.img[i][j]
                    else:
                        img[i][j] = self.img[i][j]

            # print(time.time() - start)

        else:
            size = [height, width, channels]
            img = np.zeros(size, dtype=np.uint8)

            for i in range(self.height):
                for j in range(self.width):
                    src = np.array([i, j, 1], dtype=np.int32)
                    dst = np.dot(transform_mat, src)

                    if 0 <= dst[0] < self.height:
                        if 0 <= dst[1] < self.width:
                            img[dst[0]][dst[1]] = self.img[i][j]

        # cv.cvtColor(img, cv.COLOR_RGB2BGR, img)
        # cv.imshow('blank', img)
        # if cv.waitKey(0) == 27:
        #     cv.destroyAllWindows()
        self.processed_img = img

    def rotate(self, angle, clockwise=True, cut=True):
        """Rotates the image clockwise or anti-clockwise.

        Rotate the image. Keep the full image or cutting edges.

        Args:
            angle (int): The angle of rotations.
            clockwise (bool): Clockwise or not

        Returns:
            Rotated image.
        """
        # TODO Rotate while keep the original image. Cut func.
        if not clockwise:
            angle = -angle

        rad = angle * math.pi / 180.0
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        height, width, channels = self.get_shape()
        size = [height, width, channels]
        img = np.zeros(size, dtype=np.uint8)

        trans_descartes = np.array([[-1, 0, 0],
                                    [0, 1, 0],
                                    [0.5 * height, -0.5 * width, 1]], dtype=np.float32)
        trans_back = np.array([[-1, 0, 0],
                               [0, 1, 0],
                               [0.5 * height, 0.5 * width, 1]], dtype=np.float32)
        rotate_mat = np.array([[cos_a, sin_a, 0],
                               [-sin_a, cos_a, 0],
                               [0, 0, 1]])
        trans_mat = np.dot(np.dot(trans_descartes, rotate_mat), trans_back)

        for i in range(self.height):
            for j in range(self.width):
                src = np.array([i, j, 1], dtype=np.int32)
                dst = np.dot(src, trans_mat)
                x = int(dst[0])
                y = int(dst[1])
                if 0 <= x < height and 0 <= y < width:
                    img[x][y] = self.img[i][j]

        cv.cvtColor(img, cv.COLOR_RGB2BGR, img)
        cv.imshow('blank', img)
        if cv.waitKey(0) == 27:
            cv.destroyAllWindows()


if __name__ == '__main__':
    ipg = ImageProcessor('bauhaus.jpg')
    # ipg.shift(-20, -30, cut=True)
    # print(ipg.get_shape())
    ipg.rotate(90, clockwise=False)
