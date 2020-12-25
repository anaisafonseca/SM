#Anaísa Forti da Fonseca - 11811ECP012
#Lucas Alesterio Marques Vieira - 11621ECP016
#Victoria Maria Veloso Rodrigues - 11811ECP003

from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtWidgets import QMainWindow,QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import numpy as np
import sys
import cv2
import qimage2ndarray
import imutils
from scipy import ndimage

def detect_blur_fft(image, size=60, thresh=10, vis=False):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return (mean, mean <= thresh)


def blur_detect(gray):
    gray = imutils.resize(gray, width=500)
    # apply our blur detector using the FFT
    (mean, blurry) = detect_blur_fft(gray, size=60,
                                     thresh=20, vis=-1 > 0)
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry" if blurry else "Not Blurry"
    text = text.format(mean)
    if blurry: return text, color;
    else: return text, color;

kernel = {
    'identity': np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float),
    'edge detection': np.array([[1,0,-1],[0,0,0],[-1,0,1]], dtype=float),
    'laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float),
    'laplacian w/ diagonals': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float),
    'laplacian of gaussian': np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]], dtype=float),
    'scharr': np.array([[-3, 0, 3],[-10,0,10],[-3, 0, 3]], dtype=float),
    'sobel edge horizontal': np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float),
    'sobel edge vertical': np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float),
    'line detection horizontal': np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float),
    'line detection vertical': np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=float),
    'line detection 45°': np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype=float),
    'line detection 135°': np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype=float),
    'box blur': (1/9)*np.ones((3,3), dtype=float),
    'gaussian blur 3x3': (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float),
    'gaussian blur 5x5': (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
    'sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
    'unsharp masking': (-1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
}

class window(QMainWindow):
    def __init__(self):
        super().__init__()
        window.setObjectName(self,"window")
        window.resize(self,808, 534)
        self.setWindowTitle("SM final")
        self.buttons()
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        
    def viewCam(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
        conv = ndimage.convolve(frame, kernel[self.comboBox.currentText()], mode='constant', cval=0)
        if (self.checkBox.isChecked()):
            text, color = blur_detect(conv)
            cv2.putText(conv, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2)
        image = qimage2ndarray.array2qimage(conv)
        self.label.setPixmap(QPixmap.fromImage(image))

    def controlTimer(self):
        if not self.timer.isActive():
            self.cap = cv2.VideoCapture(0)
            self.timer.start(20)

    def controlTimer2(self):
        if self.timer.isActive():
            self.timer.stop()
            self.cap.release()

    def colors(self):
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 70, 70))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 70, 70))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 70, 70))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 70, 70))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(70, 70, 70))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(34, 34, 34))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        return palette

    def buttons(self):
        self.setPalette(self.colors())
        self.setObjectName("centralwidget")
        self.startButton = QtWidgets.QPushButton("Start", self)
        self.startButton.setGeometry(QtCore.QRect(680, 10, 91, 41))
        self.startButton.setStyleSheet("background-color:rgb(0, 0, 255);\n"
                                       "color:white;\n"
                                       "border-radius:5px;\n"
                                       "font:500 15px;")
        self.startButton.setObjectName("startButton")
        self.startButton.clicked.connect(self.controlTimer)
        self.stopButton = QtWidgets.QPushButton("Stop", self)
        self.stopButton.clicked.connect(self.controlTimer2)
        self.stopButton.setGeometry(QtCore.QRect(680, 60, 91, 41))
        self.stopButton.setStyleSheet("background-color:rgb(0, 0, 255);\n"
                                      "color:white;\n"
                                      "border-radius:5px;\n"
                                      "font:500 15px;")
        self.stopButton.setObjectName("stopButton")

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 691, 441))
        self.label.setText("")
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.setGeometry(QtCore.QRect(655, 150, 140, 21))
        self.comboBox.setObjectName("comboBox")

        # Add options:
        self.comboBox.addItem("identity")
        self.comboBox.addItem("edge detection")
        self.comboBox.addItem("laplacian")
        self.comboBox.addItem("laplacian of gaussian")
        self.comboBox.addItem("scharr")
        self.comboBox.addItem("sobel edge horizontal")
        self.comboBox.addItem("sobel edge vertical")
        self.comboBox.addItem("line detection horizontal")
        self.comboBox.addItem("line detection vertical")
        self.comboBox.addItem("line detection 45°")
        self.comboBox.addItem("line detection 135°")
        self.comboBox.addItem("box blur")
        self.comboBox.addItem("gaussian blur 3x3")
        self.comboBox.addItem("gaussian blur 5x5")
        self.comboBox.addItem("sharpen")
        self.comboBox.addItem("unsharp masking")

        self.comboBox.setStyleSheet("color:black;border-radius:5px;")

        self.checkBox = QtWidgets.QCheckBox(self)
        self.checkBox.setGeometry(QtCore.QRect(690, 120, 70, 17))
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setText("Blur detect")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec_())