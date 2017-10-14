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
        self.log_left = QtGui.QLabel()
        self.log_right = QtGui.QLabel()
        self.log_mid = QtGui.QLabel()

        self.obs_text = "watch the obstacle on your "
        self.clear_text = " clear"
        self.dir_dic = {0: 'left', 1: 'right', 2: 'center'}
        self.status = [0, 0, 0]

        self.text = dict()
        for loc in range(3):
            self.text[loc] = QtGui.QLabel()
            self.text[loc].setText(self.dir_dic[loc] + self.clear_text)

        layout = QtGui.QVBoxLayout(self)
        for loc in range(3):
            layout.addWidget(self.text[loc])
        
    def set_status(self, target, status):
        if status == OBS:
            self.text[target].setText(self.obs_text + self.dir_dic[target])
        else:
            self.text[target].setText(self.dir_dic[target] + self.clear_text)


if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('Phonon Player')
    player = Player()
    window = QtGui.QWidget()
    logger = Logger()
    box = QtGui.QHBoxLayout()
    box.addWidget(player)
    box.addWidget(logger)
    window.setLayout(box)

    window.show()
    sys.exit(app.exec_())
