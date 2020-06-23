import sys
from ImageProcessor import ImageProcessor
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QToolTip, QDesktopWidget, QGridLayout
from PyQt5.QtWidgets import QMainWindow, QMenu, QAction, QToolBar
from PyQt5.QtCore import Qt


class ImgProcGUI(QWidget):
    def __init__(self):
        """Initialize Image Processor GUI"""
        super().__init__()

        self.label = QLabel()

        self.show()
        # self.btn_quit = QPushButton('Quit', self)
        # self.btn_process = QPushButton('Process', self)
        # self.btn_save = QPushButton('Save', self)
        # self.btn_open = QPushButton('Open', self)

    #     self.init_()
    #
    # def init_(self):
    #     """UI initialize"""
    #
    #
    #     # Connect signals and slots
    #     self.show()




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.ipg = ImgProcGUI()
        self.img_proc = ImageProcessor()
        self.label = QLabel()
        self.extensions = "*.png *.jpg *.jpeg *.tiff *.bmp *.gif *.tfrecords"
        self.init_()

    def init_(self):
        # Status bar
        self.statusBar().showMessage('Ready')

        # menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

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

        trans_gray_act = QAction('gray', self)
        trans_gray_act.setStatusTip('Transform to gray')
        trans_gray_act.triggered.connect(self.trans_gray)

        file_menu.addAction(open_file_act)

        self.setCentralWidget(self.label)

        # tool bar
        # open_toolbar = QToolBar('Open')
        # open_toolbar.addAction(open_file_act)
        # self.addToolBar(open_toolbar)
        # save_toolbar = QToolBar('Save')
        # save_toolbar.addAction(save_file_act)
        # self.addToolBar(save_toolbar)
        keep_toolbar = QToolBar('Save')
        keep_toolbar.addAction(save_changes_act)
        self.addToolBar(keep_toolbar)

        gray_toolbar = QToolBar('Gray')
        gray_toolbar.addAction(trans_gray_act)
        self.addToolBar(gray_toolbar)

        # self.toolbar.addAction()
        self.resize(480, 360)
        self.center()
        self.setWindowTitle('Image Processor')
        self.show()

    def load_img(self):
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
        self.show_img()

    def save_img(self):
        # Get file path to save
        fp, tmp = QFileDialog.getSaveFileName(self, caption='Save Image', directory='./',
                                              filter=self.extensions)
        if fp is '':
            return

        if self.img_proc.get_img() is None or self.img_proc.get_img().size == 1:
            return

        # Save image file by image processor
        self.img_proc.save_img(fp)

    def process_imag(self):
        pass

    def save_changes(self):
        if self.img_proc.get_img() is None:
            return
        self.img_proc.save_changes()

    def show_img(self, img=None):
        # Get image size and channels，convert opencv image representation to QImage
        if img is None:
            height, width, channels = self.img_proc.get_shape()
            bytes_per_line = 3 * width
        else:
            height, width = img.shape
            bytes_per_line = width
            channels = None

        # Create QImage
        if channels is not None:
            q_img = QImage(self.img_proc.get_img().data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            q_img = QImage(self.img_proc.get_img().data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        # Show QImage
        self.label.setPixmap(QPixmap.fromImage(q_img))
        self.center(width, height)

    def trans_gray(self):
        img = self.img_proc.trans_gray()
        self.show_img(img)


    # Override
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

        # DISCARDED METHOD
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
    # ipg = ImgProcGUI()
    main_window = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
