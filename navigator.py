from __future__ import print_function

import tensorflow as tf
import numpy as np

import argparse
import os
import sys
import time
import scipy.io as sio
from PIL import Image

from scipy import misc

from model import DeepLabResNetModel

import skvideo.io
import matplotlib.pyplot as plt
import nav_util

import thread
from multiprocessing import Queue
from imageToText import imageToText

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
        self.logger = Logger()

        self.MAX_FRAMES = 2000
        self.FRAME_RATE = 30
        self.sess, self.pred, self.ph = nav_util.module_init()

        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.logger)
        layout.addWidget(self.video, 1)
        layout.addWidget(self.button)


    def navi(self, frames):
        frame_count = frames.shape[3]
        print(frame_count)
        pick_frame = 0
        time_start = time.time()
        print(frame_count)
        while True:
            blocked_state, out = nav_util.navigation(self.sess, self.pred, self.ph, frames[:,:,:,pick_frame])
            thread.start_new_thread(self.logger.set_status, (blocked_state,))
            time_end = time.time()
            pick_frame += int((time_end - time_start) * self.FRAME_RATE)
            print(pick_frame)
            time_start = time_end
            if pick_frame > frame_count:
                break
            pass

    def handleButton(self):
        if self.media.state() == Phonon.PlayingState:
            self.media.stop()
        else:
            path = QtGui.QFileDialog.getOpenFileName(self, self.button.text())
            if path:
                self.media.setCurrentSource(Phonon.MediaSource(path))
                cap = skvideo.io.VideoCapture(str(path), frameSize=(1080,1920))
                frame_count = 0
                all_frames = np.zeros([256, 256, 3, self.MAX_FRAMES])
                while cap.isOpened():
                    ret, img = cap.read()
                    if ret:
                        all_frames[:,:,:,frame_count] = misc.imresize(img, [256, 256])
                        frame_count += 1
                    else:
                        break
                all_frames = all_frames[:,:,:,0:frame_count]
                thread.start_new_thread(self.navi, (all_frames,))
                start = time.time()
                end = start
                while end - start < 2:
                    end = time.time()
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
        self.text_dic[6] = 'Left and front blocked!'
        self.text_dic[5] = 'Left and right blocked!'
        self.text_dic[4] = 'Left blocked!'
        self.text_dic[3] = 'Front and right blocked!'
        self.text_dic[2] = 'Front blocked!'
        self.text_dic[1] = 'Right blocked!'
        self.text_dic[0] = 'All clear!'

        
        self.red = "<P><i><FONT COLOR='#ff0000' FONT SIZE = 5>"
        self.green = "<P><i><FONT COLOR='#00ff00' FONT SIZE = 5>"
        self.log.setText('<P><b><FONT SIZE = 5>Please Select a Video')
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
        box = QtGui.QVBoxLayout()
        box.addWidget(self.player)
        self.setLayout(box)



if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('the Light')
    nav = Navigator()
    nav.show()
    sys.exit(app.exec_())
