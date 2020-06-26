import sys
from ImageProcessor import ImageProcessor
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QMessageBox, QLineEdit, QTextEdit, \
    QComboBox, QDialog, QDialogButtonBox, QVBoxLayout, QCheckBox
from PyQt5.QtWidgets import QToolTip, QDesktopWidget, QGridLayout
from PyQt5.QtWidgets import QMainWindow, QMenu, QAction, QToolBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QRegExpValidator, QDoubleValidator
from PyQt5.QtCore import QRegExp


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.img_proc = ImageProcessor()
        self.label = QLabel()
        self.extensions = "*.png *.jpg *.jpeg *.tiff *.bmp *.gif *.tfrecords"
        self.info = None
        self.init_()

    def init_(self):
        # Status bar
        self.statusBar().showMessage('Ready')

        # menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        info_menu = menubar.addMenu('Get Info')
        shape_menu = menubar.addMenu('Shape')
        gray_menu = menubar.addMenu('Process')

        # Actions
        open_file_act = QAction("Open", self)
        open_file_act.setStatusTip("Open file")
        open_file_act.triggered.connect(self.load_img)

        save_file_act = QAction("Save", self)
        save_file_act.setStatusTip("Save file")
        save_file_act.triggered.connect(self.save_img)

        save_changes_act = QAction('Save changes', self)
        save_changes_act.setStatusTip('Save changes')
        save_changes_act.triggered.connect(self.save_changes)

        trans_gray_act = QAction('Change gray level', self)
        trans_gray_act.setStatusTip('Change gray level')
        trans_gray_act.triggered.connect(self.trans_gray)

        get_info_act = QAction('Info', self)
        get_info_act.setStatusTip('Get info of image')
        get_info_act.triggered.connect(self.get_info)

        get_pixel_color = QAction('Pixel color', self)
        get_pixel_color.setStatusTip('Get pixel color of image')
        get_pixel_color.triggered.connect(self.get_pixel_color)

        hist_act = QAction('Show hist', self)
        hist_act.setStatusTip('Show histogram')
        hist_act.triggered.connect(self.show_hist)

        resize_act = QAction('Resize', self)
        resize_act.setStatusTip('Resize image')
        resize_act.triggered.connect(self.resize_img)

        rotate_act = QAction('Rotate', self)
        rotate_act.setStatusTip('Rotate image')
        rotate_act.triggered.connect(self.rotate_img)

        shift_act = QAction('Shift', self)
        shift_act.setStatusTip('Shift image')
        shift_act.triggered.connect(self.shift_img)

        hist_equalization_act = QAction('Hist equalization', self)
        hist_equalization_act.setStatusTip('Hist equalization')
        hist_equalization_act.triggered.connect(self.hist_equalization)

        smooth_act = QAction('Smooth', self)
        smooth_act.setStatusTip('Smooth')
        smooth_act.triggered.connect(self.smooth)

        sharpen_act = QAction('Sharpen', self)
        sharpen_act.setStatusTip('Sharpen')
        sharpen_act.triggered.connect(self.sharpen)

        # Add actions
        file_menu.addAction(open_file_act)
        file_menu.addAction(save_file_act)
        info_menu.addAction(get_info_act)
        info_menu.addAction(get_pixel_color)
        info_menu.addAction(hist_act)
        gray_menu.addAction(trans_gray_act)
        gray_menu.addAction(hist_equalization_act)
        gray_menu.addAction(smooth_act)
        gray_menu.addAction(sharpen_act)
        shape_menu.addAction(resize_act)
        shape_menu.addAction(rotate_act)
        shape_menu.addAction(shift_act)

        # Label for displaying image
        self.setCentralWidget(self.label)

        # tool bar
        keep_toolbar = QToolBar('Save')
        keep_toolbar.addAction(save_changes_act)
        self.addToolBar(keep_toolbar)

        # Set window position and size
        self.resize(480, 360)
        self.center()
        self.setWindowTitle('Image Processor')
        self.show()

    def status_msg(self, msg, sec=None):
        if sec is None:
            self.statusBar().showMessage(msg)
        else:
            self.statusBar().showMessage(msg, msecs=sec * 1000)

    def status_clear(self):
        self.statusBar().clearMessage()

    def load_img(self):
        self.status_msg('Opening file...')
        # Load image from disk
        fp, tmp = QFileDialog.getOpenFileName(self, caption='Open Image', directory='./',
                                              filter=self.extensions)

        if fp is '':
            return

        # Use Image Processor class to load image file
        self.img_proc.load_img(fp)

        if self.img_proc.get_img() is None or self.img_proc.get_img().size == 1:
            return

        # Show image
        self.show_img(center=True)
        self.status_msg('Done!', 3)

    def save_img(self):
        self.status_msg('Saving file...')
        # Get file path to save
        fp, tmp = QFileDialog.getSaveFileName(self, caption='Save Image', directory='./',
                                              filter=self.extensions)
        if fp is '':
            return

        if self.img_proc.get_img() is None or self.img_proc.get_img().size == 1:
            return

        # Save image file by image processor
        self.img_proc.save_img(fp)
        self.status_msg('Done!', 3)

    def get_info(self):
        self.status_msg('Getting info...')
        height, width, channels = self.img_proc.get_shape()
        self.info = Info(height, width, channels)
        self.info.resize(200, 200)
        self.info.show()
        self.center(self.info)
        self.status_msg('Done!', 3)

    def show_hist(self):
        self.status_msg('Showing histogram...')
        hist = self.img_proc.get_hist()
        if hist is None:
            self.status_clear()
            return
        self.hist = Hist(hist)
        self.hist.resize(256, 256)
        self.hist.show()
        self.center(self.hist)
        self.status_msg('Done!', 3)

    def get_pixel_color(self):
        self.status_msg('Getting pixel color...')
        if self.img_proc.get_img() is None:
            self.status_clear()
            return
        max_h, max_v, _ = self.img_proc.get_shape()
        dialog = PixelPos(max_h - 1, max_v - 1)
        result = dialog.exec_()
        pos = dialog.pos()
        if pos[0] is '' or pos[1] is '' or pos[2] is '':
            self.status_clear()
            return
        r, g, b = self.img_proc.get_pixel_color(int(pos[0]), int(pos[1]))
        self.pixel = PixelColor(r, g, b)
        self.pixel.resize(200, 200)
        self.pixel.show()
        self.center(self.pixel)
        self.status_msg('Done!', 3)

    def save_changes(self):
        if self.img_proc.get_img() is None:
            self.status_clear()
            return

        self.status_msg('Saving changes...')
        if self.img_proc.get_img() is None:
            self.status_msg('Nothing to save!')
            return
        self.img_proc.save_changes()
        self.status_msg('Done!', 3)

    def show_img(self, img=None, center=False):
        # Get image size and channels，convert opencv image representation to QImage
        if img is None:
            img = self.img_proc.get_img()

        height, width, channels = img.shape
        bytes_per_line = channels * width

        # Create QImage
        if channels is 1:
            img = img.squeeze()
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            q_img = QImage(self.img_proc.get_img().data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Show QImage
        self.label.setPixmap(QPixmap.fromImage(q_img))
        if center:
            self.center(x_offset=width, y_offset=height)

    def trans_gray(self):
        self.status_msg('Transforming to gray...')

        if self.img_proc.get_img() is None:
            self.status_clear()
            return

        dialog = GrayProcess()
        result = dialog.exec_()
        level = dialog.level()

        _, _, channels = self.img_proc.get_shape()
        if channels is 3:
            reply = QMessageBox.question(self, 'Warning',
                                         "This process will change the image to gray scale image, continue?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                return
        img = self.img_proc.trans_gray(level=int(level))
        if img is None:
            self.status_clear()
            return
        self.show_img(img)
        self.status_msg('Done!', 3)

    def resize_img(self):
        self.status_msg('Resizing image...')
        if self.img_proc.get_img() is None:
            self.status_clear()
            return

        dialog = ResizeFactor()
        result = dialog.exec_()
        size = dialog.factor()
        if size[0] is '' or size[1] is '' or size[0] is '0' or size[1] is '0':
            self.status_clear()
            return

        img = self.img_proc.resize(float(size[0]), float(size[1]))
        self.show_img(img)
        self.status_msg('Done!', 3)

    def rotate_img(self):
        self.status_msg('Rotating image...')
        if self.img_proc.get_img() is None:
            self.status_clear()
            return

        dialog = Angle()
        result = dialog.exec_()
        angle = dialog.angle()
        cut = dialog.cut()
        if angle is '':
            self.status_clear()
            return

        img = self.img_proc.rotate(int(angle), cut=cut)
        self.show_img(img)
        self.status_msg('Done!', 3)

    def shift_img(self):
        self.status_msg('Shifting image...')
        if self.img_proc.get_img() is None:
            self.status_clear()
            return

        dialog = Shift()
        result = dialog.exec_()
        offset = dialog.offset()
        cut = dialog.cut()
        if offset[0] is '' or offset[1] is '':
            self.status_clear()
            return

        img = self.img_proc.shift(int(offset[0]), int(offset[1]),cut=cut)
        self.show_img(img)
        self.status_msg('Done!', 3)

    def hist_equalization(self):
        self.status_msg('Hist equalization...')

        if self.img_proc.get_img() is None:
            self.status_clear()
            return

        _, _, channels = self.img_proc.get_shape()
        if channels is 3:
            reply = QMessageBox.question(self, 'Warning',
                                         "This process will change the image to gray scale image, continue?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                return
        img = self.img_proc.hist_equalization()
        if img is None:
            self.status_clear()
            return
        self.show_img(img)
        self.status_msg('Done!', 3)

    def smooth(self):
        self.status_msg('Smoothing...')

        if self.img_proc.get_img() is None:
            self.status_clear()
            return

        _, _, channels = self.img_proc.get_shape()
        if channels is 3:
            reply = QMessageBox.question(self, 'Warning',
                                         "This process will change the image to gray scale image, continue?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                return
        img = self.img_proc.smooth()
        if img is None:
            self.status_clear()
            return
        self.show_img(img)
        self.status_msg('Done!', 3)

    def sharpen(self):
        self.status_msg("Sobel operator's gradient, sharpening...")

        if self.img_proc.get_img() is None:
            self.status_clear()
            return

        _, _, channels = self.img_proc.get_shape()
        if channels is 3:
            reply = QMessageBox.question(self, 'Warning',
                                         "This process will change the image to gray scale image, continue?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.No:
                return
        img = self.img_proc.sharpen()
        if img is None:
            self.status_clear()
            return
        self.show_img(img)
        self.status_msg('Done!', 3)


    # Override
    def closeEvent(self, event):
        """Pop up a warning message before quit"""
        reply = QMessageBox.question(self, 'Warning',
                                     "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def center(self, window=None, x_offset=None, y_offset=None):
        """Center the window"""
        if window is None:
            window = self
        qrect_screen = QDesktopWidget().screenGeometry()
        qrect_self = window.geometry()
        if x_offset is not None and y_offset is not None:
            window.move((qrect_screen.width() - x_offset) / 2, (qrect_screen.height() - y_offset) / 2)
        else:
            window.move((qrect_screen.width() - qrect_self.width()) / 2,
                        (qrect_screen.height() - qrect_self.height()) / 2)


class Info(QWidget):
    def __init__(self, height, width, channels):
        super().__init__()
        if height is None or width is None or channels is None:
            warning = QLabel(self)
            warning.setText("No Info!")
            warning.setAlignment(Qt.AlignCenter)
            return
        self.lbl_height = QLabel('Height:')
        self.height = QLabel(str(height))
        self.lbl_width = QLabel('Width:')
        self.width = QLabel(str(width))
        self.lbl_channels = QLabel('Channels')
        self.channels = QLabel(str(channels))
        self.init_()

    def init_(self):
        grid = QGridLayout()
        grid.addWidget(self.lbl_height, 1, 0)
        grid.addWidget(self.height, 1, 1)
        grid.addWidget(self.lbl_width, 2, 0)
        grid.addWidget(self.width, 2, 1)
        grid.addWidget(self.lbl_channels, 3, 0)
        grid.addWidget(self.channels, 3, 1)
        self.setLayout(grid)


class PixelColor(QWidget):
    def __init__(self, r, g, b):
        super().__init__()
        if r is None or g is None or b is None:
            warning = QLabel(self)
            warning.setText("No Info!")
            warning.setAlignment(Qt.AlignCenter)
            return
        self.lbl_height = QLabel('Red:')
        self.height = QLabel(str(r))
        self.lbl_width = QLabel('Green:')
        self.width = QLabel(str(g))
        self.lbl_channels = QLabel('Blue:')
        self.channels = QLabel(str(b))
        self.init_()

    def init_(self):
        grid = QGridLayout()
        grid.addWidget(self.lbl_height, 1, 0)
        grid.addWidget(self.height, 1, 1)
        grid.addWidget(self.lbl_width, 2, 0)
        grid.addWidget(self.width, 2, 1)
        grid.addWidget(self.lbl_channels, 3, 0)
        grid.addWidget(self.channels, 3, 1)
        self.setLayout(grid)

class Hist(QWidget):
    def __init__(self, img):
        super().__init__()
        self.lbl = QLabel(self)
        if img is None:
            return

        height, width, channels = img.shape
        bytes_per_line = channels * width

        # Create QImage
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Show QImage
        self.lbl.setPixmap(QPixmap.fromImage(q_img))


class GrayProcess(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('choose level')
        self.lbl_height = QLabel('Level:')
        items = ['2', '4', '8', '16', '32', '64', '128', '256']
        self.comboBox = QComboBox()
        self.comboBox.addItems(items)
        self.comboBox.setCurrentIndex(7)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.init_()

    def init_(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_height)
        vbox.addWidget(self.comboBox)
        vbox.addWidget(self.buttons)
        self.setLayout(vbox)

    def level(self):
        return self.comboBox.currentText()

    @staticmethod
    def get_level():
        dialog = GrayProcess()
        result = dialog.exec_()
        level = dialog.level()
        return level


class PixelPos(QDialog):
    def __init__(self, max_h, max_w):
        super().__init__()
        self.setWindowTitle('Choose position')
        self.lbl_height = QLabel('Height: (max ' + str(max_h) + ')')
        self.lbl_width = QLabel('Width: (max ' + str(max_w) + ')')

        height_va = QIntValidator(0, max_h)
        self.height_in = QLineEdit(self)
        self.height_in.setValidator(height_va)

        width_va = QIntValidator(0, max_w)
        self.width_in = QLineEdit(self)
        self.width_in.setValidator(width_va)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.init_()

    def init_(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_height)
        vbox.addWidget(self.height_in)
        vbox.addWidget(self.lbl_width)
        vbox.addWidget(self.width_in)
        vbox.addWidget(self.buttons)
        self.setLayout(vbox)

    def pos(self):
        return self.height_in.text(), self.width_in.text()


class ResizeFactor(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Resize Factor')
        self.lbl_height = QLabel('Height: (max 10)')
        self.lbl_width = QLabel('Width: (max 10)')

        rx = QRegExp('^(10|[0-9](\.[0-9])?)$')
        height_va = QRegExpValidator(rx)
        self.height_in = QLineEdit(self)
        self.height_in.setValidator(height_va)

        width_va = QRegExpValidator(rx)
        self.width_in = QLineEdit(self)
        self.width_in.setValidator(width_va)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.init_()

    def init_(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_height)
        vbox.addWidget(self.height_in)
        vbox.addWidget(self.lbl_width)
        vbox.addWidget(self.width_in)
        vbox.addWidget(self.buttons)
        self.setLayout(vbox)

    def factor(self):
        return self.height_in.text(), self.width_in.text()


class Angle(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Choose angle')
        self.lbl_height = QLabel('Agnle: (0 - 360)')

        height_va = QIntValidator(0, 360)
        self.height_in = QLineEdit(self)
        self.height_in.setValidator(height_va)

        self.cb = QCheckBox("Cut image?", self)
        self.cb.setChecked(True)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.init_()

    def init_(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_height)
        vbox.addWidget(self.height_in)
        vbox.addWidget(self.cb)
        vbox.addWidget(self.buttons)
        self.setLayout(vbox)

    def angle(self):
        return self.height_in.text()

    def cut(self):
        return self.cb.isChecked()


class Shift(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Shift offset')
        self.lbl_height = QLabel('Height: (pixels)')
        self.lbl_width = QLabel('Width: (pixels)')

        rx = QRegExp('^-?[0-9]+$')
        va = QRegExpValidator(rx)

        self.height_in = QLineEdit(self)
        self.height_in.setValidator(va)


        self.width_in = QLineEdit(self)
        self.width_in.setValidator(va)

        self.cb = QCheckBox("Cut image?", self)
        self.cb.setChecked(True)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.init_()

    def init_(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.lbl_height)
        vbox.addWidget(self.height_in)
        vbox.addWidget(self.lbl_width)
        vbox.addWidget(self.width_in)
        vbox.addWidget(self.cb)
        vbox.addWidget(self.buttons)
        self.setLayout(vbox)

    def offset(self):
        return self.height_in.text(), self.width_in.text()

    def cut(self):
        return self.cb.isChecked()


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
