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
        """Load image from disk

        Args:
            fp (str): Path to image file
        """
        self.img = cv.imread(fp)
        cv.cvtColor(self.img, cv.COLOR_BGR2RGB, self.img)
        self.processed_img = self.img
        self.update_img_property()

    def update_img_property(self):
        """Update image properties, including height, width and channels"""
        self.height, self.width, self.channels = self.img.shape

    def get_img(self):
        """Get image"""
        return self.processed_img

    def set_img(self, img):
        """Set given image to class image"""
        self.img = img

    def restore_changes(self):
        """Restore changes of image"""
        self.processed_img = self.img

    def save_changes(self):
        """Save changes on the processed image"""
        self.img = self.processed_img
        self.update_img_property()

    def save_img(self, fp):
        """Save the image to disk"""
        cv.cvtColor(self.img, cv.COLOR_BGR2RGB, self.img)
        cv.imwrite(fp, self.img)

    def show(self, img=None, name='Image'):
        """Display image. Press 'esc' to exit.

        Args:
            img (numpy.array): Image array representation.
            name (str): Name of the window.
        """
        if img is None:
            img = self.img

        cv.cvtColor(img, cv.COLOR_RGB2BGR, img)
        cv.imshow(name, img)
        if cv.waitKey(0) == 27:
            cv.destroyAllWindows()

    def get_split_color(self, mode):
        """Split image color

        Args:
            mode (str): b - blue; r - red; g - green.

        Returns:
            Single channel image.
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

        self.processed_img = img
        return img

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

    def shift(self, x, y, cut=False):
        """Shift the image vertically and horizontally.

        If cut the shifted image, the part shifted out will not
        appear and the image size remain the same. If not cut the
        image, the blank area will be filled with black. Size of
        image will increase.

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
            img = self.create_blank_img(height + abs(x), width + abs(y))

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
            img = self.create_blank_img()

            for i in range(self.height):
                for j in range(self.width):
                    src = np.array([i, j, 1], dtype=np.int32)
                    dst = np.dot(transform_mat, src)

                    if 0 <= dst[0] < self.height:
                        if 0 <= dst[1] < self.width:
                            img[dst[0]][dst[1]] = self.img[i][j]


        self.processed_img = img
        return img

    def rotate(self, angle, clockwise=True, cut=True):
        """Rotates the image clockwise or anti-clockwise.

        Rotate the image. Keep the full image or cutting edges.

        Args:
            angle (int): The angle of rotations.
            clockwise (bool): Clockwise or not.
            cut (bool): If rotation cutting the image or not.

        Returns:
            Rotated image.
        """
        if not clockwise:
            angle = -angle

        rad = angle * math.pi / 180.0
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        height, width, channels = self.get_shape()

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

        if cut:
            img = self.create_blank_img()
            for i in range(self.height):
                for j in range(self.width):
                    src = np.array([i, j, 1], dtype=np.int32)
                    dst = np.dot(src, trans_mat)
                    x = int(dst[0])
                    y = int(dst[1])
                    if 0 <= x < height and 0 <= y < width:
                        img[x][y] = self.img[i][j]
        else:
            org_x1 = np.array([0.5 * height, -0.5 * width, 1], dtype=np.int32)
            org_x2 = np.array([-0.5 * height, -0.5 * width, 1], dtype=np.int32)

            new_x1 = np.dot(org_x1, rotate_mat)
            new_x2 = np.dot(org_x2, rotate_mat)

            new_height = 2 * math.ceil(max(abs(new_x1[0]), abs(new_x2[0])))
            new_width = 2 * math.ceil(max(abs(new_x1[1]), abs(new_x2[1])))
            img = self.create_blank_img(new_height + 1, new_width + 1)
            new_trans_back = np.array([[-1, 0, 0],
                                       [0, 1, 0],
                                       [0.5 * new_height, 0.5 * new_width, 1]], dtype=np.float32)
            new_trans_mat = np.dot(np.dot(trans_descartes, rotate_mat), new_trans_back)
            for i in range(self.height):
                for j in range(self.width):
                    src = np.array([i, j, 1], dtype=np.int32)
                    dst = np.dot(src, new_trans_mat)
                    x = int(dst[0])
                    y = int(dst[1])
                    img[x][y] = self.img[i][j]

        self.processed_img = img
        return img

    def resize(self, m, n):
        height, width, channels = self.get_shape()
        height = int(height * m)
        width = int(width * n)
        img = self.create_blank_img(height, width, channels)
        for i in range(height):
            for j in range(width):
                src_i = int(i / m)
                src_j = int(j / n)
                img[i][j] = self.img[src_i][src_j]

        self.processed_img = img
        return img

    def trans_gray(self, level=256):
        """Transform an RGB image to Gray Scale image.

        Gray scale can be quantized to 256, 128, 64, 32,
        16, 8, 4, 2 levels.

        Args:
            level (int): Quantization level. Default 256.

        Returns:
            Gray scale image.
        """
        if self.channels is 1:
            return self.get_img()

        n = math.log2(level)
        if n < 1 or n > 8:
            raise ValueError('Quantization level wrong! Must be exponential value of 2')

        # Turn image from RGB to Gray scale image
        img = self.create_blank_img(channels=1)
        step = 256 / level

        for i in range(self.height):
            for j in range(self.width):
                pixel = self.img[i][j]
                gray = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
                mapped_gray = int(gray / step) / (level - 1) * 255
                img[i][j] = round(mapped_gray)

        self.processed_img = img
        return img

    def create_blank_img(self, height=None, width=None, channels=3):
        """Create a blank pure black image.

        Default create a blank black image with same height,
        width and channels as the loaded image.

        Args:
            height (int): Height of new image. Measured by pixels.
            width (int): Width of new image. Measured by pixels.
            channels (int): Channels. Default 3, RGB.

        Returns:
            New image.
        """
        if not height and not width:
            height, width, _ = self.get_shape()
        if not height or not width:
            raise Exception("Invalid height or width!")

        size = (height, width, channels)
        img = np.zeros(size, dtype=np.uint8)
        return img

    def get_hist(self):
        """Get histogram of given image

        Returns:
            Image of histogram of the image.
        """
        hist = np.zeros(256, dtype=np.uint32)
        hist_img = np.zeros((256, 256, 3), dtype=np.uint8)
        img = self.trans_gray()

        for i in range(self.height):
            for j in range(self.width):
                # print(img[i][j][0])
                g_p = int(img[i][j])
                hist[g_p] += 1

        # Maximum count in all 256 levels
        max_freq = max(hist)

        for i in range(256):
            x = (i, 255)
            # Calculate the relative frequency compared to maximum frequency
            p = int(255 - hist[i] * 255 / max_freq)
            y = (i, p)
            cv.line(hist_img, x, y, (0, 255, 0))

        return hist_img

    def hist_equalization(self):
        """Histogram equalization of the image.

        Returns:
            Image after histogram equalization.
        """
        hist = np.zeros(256, dtype=np.uint32)
        img = self.trans_gray()

        for i in range(self.height):
            for j in range(self.width):
                g_p = int(img[i][j])
                hist[g_p] += 1

        hist_c = np.zeros(256, dtype=np.uint32)
        hist_c[0] = hist[0]
        for i in range(1, 256):
            hist_c[i] = hist_c[i-1] + hist[i]

        factor = 255.0 / (self.height * self.width)
        for i in range(self.height):
            for j in range(self.width):
                g_p = int(img[i][j])
                g_q = int(factor * hist_c[g_p])
                img[i][j] = g_q

        self.processed_img = img
        return img

    def smooth(self):
        height = self.height
        width = self.width
        img = self.trans_gray()
        filtered_img = self.create_blank_img()
        # TODO h= np.array()

        for i in range(height):
            for j in range(width):
                if i in [0, height-1] or j in [0, width-1]:
                    filtered_img[i][j] = img[i][j]
                else:
                    x = img[i-1:i+1, j-1:j+1]
                    # TODO filtered_img[i][j] = np.dot(x, )
        pass


if __name__ == '__main__':
    ipg = ImageProcessor('bauhaus.jpg')
    # ipg.shift(-20, -30, cut=True)
    # print(ipg.get_shape())
    # ipg.rotate(90, clockwise=False, cut=True)
    # ipg.resize(m=0.5, n=2)
    # ipg.show(ipg.trans_gray())
    # ipg.show(ipg.get_hist())
    # ipg.show(ipg.hist_equalization())
    ipg.smooth()


