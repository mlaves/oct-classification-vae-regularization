#!/usr/bin/env python3

import sys
import torch
import numpy as np
from models import VariationalAutoEncoderModelShort
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QLabel, QSlider, QWidget, QMainWindow, QGridLayout


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HelloWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setWindowTitle("Latent space")

        widget = QWidget(self)
        self.setCentralWidget(widget)

        gridLayout = QGridLayout()
        widget.setLayout(gridLayout)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximum(255)
        self.slider.setMinimum(-255)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.valuechange)

        gridLayout.addWidget(self.label, 0, 0)
        gridLayout.addWidget(self.slider, 1, 0)

        img = np.zeros((256, 256), dtype=np.uint8)
        image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap(image)

        self.label.setPixmap(pix)

        # create a model
        self.net = VariationalAutoEncoderModelShort(num_classes=4, latent_size=32).to(device)

        try:
            filename = "./snapshots/vae_best_ls32.pth.tar"
            checkpoint = torch.load(filename, map_location=device)
            pretrained_dict = checkpoint['state_dict']
            self.net.load_state_dict(pretrained_dict)
            print("Loading pretrained weights from " + filename)
        except FileNotFoundError as e:
            print(e)
            exit(-1)

        self.net.eval()

    def valuechange(self):
        with torch.no_grad():
            y = torch.FloatTensor([[2.4870, -0.6960, -0.4333, -0.7585]]).to(device)
            z = torch.rand(1, 32).to(device).log()

            x_rec = self.net.decoder(y, z)
            img = x_rec.data.cpu().numpy()[0]
            img = img * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0).copy()

            image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap(image)

            self.label.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = HelloWindow()
    mainWin.show()
    sys.exit(app.exec_())
