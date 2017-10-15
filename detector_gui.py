from navigator import *
from PyQt4 import QtGui, QtCore
from object_detection.detector import *
import os
import nav_util

class Detector(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.text = QtGui.QLabel()
        self.text.setText("Detector")
        self.original = QtGui.QLabel()
        self.original.setPixmap(QtGui.QPixmap("./logo_resized.png"))
        self.hbox = QtGui.QHBoxLayout()
        self.hbox.addWidget(self.original)
        self.button = QtGui.QPushButton('Choose File', self)
        self.button.clicked.connect(self.handleButton)
        
        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.text)
        vbox.addLayout(self.hbox)
        vbox.addWidget(self.button)

    def handleButton(self):
        path = QtGui.QFileDialog.getOpenFileName(self, self.button.text())
        if path:
            path = str(path)
            os.chdir('./object_detection')
            self.original.setPixmap(QtGui.QPixmap(path))
            ret_left, ret_mid, ret_right = detect(path)
            os.chdir('../')
            message = ""
            for item in ret_left:
                message += 'There are {} {} on the left-hand side.'.format(item[0], item[1])
                message += '\n'
            for item in ret_mid:
                message += 'There are {} {} in the middle.'.format(item[0], item[1])
                message += '\n'
            for item in ret_right:
                message += 'There are {} {} on the right-hand side.'.format(item[0], item[1])
                message += '\n'
            self.text.setText(message)
            for item in message.split('\n'):
                nav_util.speak(item)
                

if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('the Light')
    nav = Detector()
    nav.show()
    sys.exit(app.exec_())
