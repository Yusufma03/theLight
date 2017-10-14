from navigator import *
from PyQt4 import QtGui, QtCore

class Menu(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.navigator = Navigator()
        self.navigator_button = QtGui.QPushButton('Navigator', self)
        self.navigator_button.setText('Navigator')
        self.navigator_button.clicked.connect(self.handle_navigator)
        layout = QtGui.QHBoxLayout(self)
        layout.addWidget(self.navigator_button)

    def handle_navigator(self):
        self.navigator.show()

if __name__=='__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('the Light')
    menu = Menu()
    menu.show()
    sys.exit(app.exec_())