from PyQt4 import QtGui, QtCore
from PyQt4.phonon import Phonon
import pdb

LEFT = 0
RIGHT = 1
CENTER = 2

CLEAR = 0
OBS = 1

class Player(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.media = Phonon.MediaObject(self)
        self.media.stateChanged.connect(self.handleStateChanged)
        self.video = Phonon.VideoWidget(self)
        self.video.setMinimumSize(400, 400)
        self.audio = Phonon.AudioOutput(Phonon.VideoCategory, self)
        Phonon.createPath(self.media, self.audio)
        Phonon.createPath(self.media, self.video)
        self.button = QtGui.QPushButton('Choose File', self)
        self.button.clicked.connect(self.handleButton)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.video, 1)
        layout.addWidget(self.button)

    def handleButton(self):
        if self.media.state() == Phonon.PlayingState:
            self.media.stop()
        else:
            path = QtGui.QFileDialog.getOpenFileName(self, self.button.text())
            if path:
                self.media.setCurrentSource(Phonon.MediaSource(path))
                self.media.play()

    def handleStateChanged(self, newstate, oldstate):
        if newstate == Phonon.PlayingState:
            self.button.setText('Stop')
        elif (newstate != Phonon.LoadingState and
              newstate != Phonon.BufferingState):
            self.button.setText('Choose File')
            if newstate == Phonon.ErrorState:
                source = self.media.currentSource().fileName()
                print ('ERROR: could not play:', source.toLocal8Bit().data())
                print ('  %s' % self.media.errorString().toLocal8Bit().data())

class Logger(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.log = QtGui.QLabel()
        self.text_dic = dict()
        self.text_dic[7] = 'Slow Down! All directions blocked!'
        self.text_dic[6] = 'Left and center blocked!'
        self.text_dic[5] = 'Left and right blocked!'
        self.text_dic[4] = 'Left blocked!'
        self.text_dic[3] = 'Center and right blocked!'
        self.text_dic[2] = 'Center blocked!'
        self.text_dic[1] = 'Right blocked!'
        self.text_dic[0] = 'All clear!'

        
        self.red = "<P><i><FONT COLOR='#ff0000' FONT SIZE = 5>"
        self.green = "<P><i><FONT COLOR='#00ff00' FONT SIZE = 5>"
        self.log.setText(self.green + self.text_dic[0])
        self.log.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtGui.QHBoxLayout(self)
        layout.addWidget(self.log)

    def set_status(self, status):
        num = status[0]*4 + status[1]*2 + status[2]*1
        if num == 0:
            self.log.setText(self.green + self.text_dic[num])
            self.log.setAlignment(QtCore.Qt.AlignCenter)
        else:
            self.log.setText(self.red + self.text_dic[num])
            self.log.setAlignment(QtCore.Qt.AlignCenter)

class Navigator(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.player = Player()
        self.logger = Logger()
        box = QtGui.QVBoxLayout()
        box.addWidget(self.logger)
        box.addWidget(self.player)
        self.setLayout(box)

if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('the Light')
    nav = Navigator()
    nav.show()
    sys.exit(app.exec_())
