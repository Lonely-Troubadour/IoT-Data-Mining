import sys
from ImageProcessor import ImageProcessor
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QLabel, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QToolTip, QDesktopWidget


class ImgProcGUI(QWidget):
    def __init__(self):
        """Initialize Image Processor GUI"""
        super().__init__()
        self.img_proc = ImageProcessor()
        self.label = QLabel()
        self.btn_quit = QPushButton('Quit', self)
        self.btn_process = QPushButton('Process', self)
        self.btn_save = QPushButton('Save', self)
        self.btn_open = QPushButton('Open', self)
        self.extentions = "*.png *.jpg *.jpeg *.tiff *.bmp *.gif *.tfrecords"
        self.initUI()

    def initUI(self):
        """UI initalize"""
        # Layout
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 1, 3, 4)
        layout.addWidget(self.btn_open, 4, 1, 1, 1)
        layout.addWidget(self.btn_save, 4, 2, 1, 1)
        layout.addWidget(self.btn_process, 4, 3, 1, 1)
        layout.addWidget(self.btn_quit, 4, 4, 1, 1)

        # Connect signals and slots
        self.btn_open.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save_img)
        self.btn_process.clicked.connect(self.proccess_img)
        self.btn_quit.clicked.connect(self.close)

        self.resize(480, 360)
        self.center()
        self.setWindowTitle('Image Processor')
        self.show()

    def load_image(self):
        # Load image from disk
        fp, tmp = QFileDialog.getOpenFileName(self, caption='Open Image', directory='./',
                                              filter=self.extentions)

        if fp is '':
            return

        # Use Image Processor class to load image file
        self.img_proc.load_img(fp)

        if self.img_proc.get_img() is None or self.img_proc.get_img().size == 1:
            return

        # Show image
        self.show_img()

    def save_img(self):
        # Get file path to save
        fp, tmp = QFileDialog.getSaveFileName(self, caption='Save Image', directory='./',
                                              filter=self.extentions)
        if fp is '':
            return

        if self.img_proc.get_img() is None or self.img_proc.get_img().size == 1:
            return

        # Save image file by image processor
        self.img_proc.save_img(fp)

    def proccess_img(self):
        pass

    def show_img(self):
        # Get image size and channels，convert opencv image representation to QImage
        height, width, channels = self.img_proc.get_shape()
        bytes_perline = 3 * width

        # Create QImage
        q_img = QImage(self.img_proc.get_img().data, width, height, bytes_perline, QImage.Format_RGB888)
        # Show QImage
        self.label.setPixmap(QPixmap.fromImage(q_img))
        self.center(width, height)

    # Overried
    # def closeEvent(self, event):
    #     """Pop up a warning message before quit"""
    #     reply = QMessageBox.question(self, 'Warning',
    #                                  "Are you sure to quit?", QMessageBox.Yes |
    #                                  QMessageBox.No, QMessageBox.No)
    #
    #     if reply == QMessageBox.Yes:
    #         event.accept()
    #     else:
    #         event.ignore()

    def center(self, x_offset=None, y_offset=None):
        """Center the window"""

        # DIACARDED METHOD
        # qrect = self.frameGeometry()
        # center_point = QDesktopWidget().availableGeometry().center()
        # qrect.moveCenter(center_point)
        # self.move(qrect.topLeft())

        qrect_screen = QDesktopWidget().screenGeometry()
        qrect_self = self.geometry()
        if x_offset is not None and y_offset is not None:
            self.move((qrect_screen.width() - x_offset) / 2, (qrect_screen.height() - y_offset) / 2)
        else:
            self.move((qrect_screen.width() - qrect_self.width()) / 2,
                      (qrect_screen.height() - qrect_self.height()) / 2)


def main():
    app = QApplication(sys.argv)
    ipg = ImgProcGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
