from navigator import *
from PyQt4 import QtGui, QtCore
import speech_recognition as sr
from detector_gui import *



class Menu(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.logo = QtGui.QLabel()
        self.logo.setPixmap(QtGui.QPixmap("./logo_resized.png"))
        self.navigator = Navigator()
        self.navigator_button = QtGui.QPushButton('Navigator', self)
        self.navigator_button.setText('Navigator')
        self.navigator_button.clicked.connect(self.handle_navigator)
        self.detector_button = QtGui.QPushButton('Detector', self)
        self.detector_button.setText('Detector')
        self.detector_button.clicked.connect(self.handle_detector)
        self.detector = Detector()
        layout = QtGui.QVBoxLayout(self)
        hbox = QtGui.QHBoxLayout()
        layout.addWidget(self.logo)
        hbox.addWidget(self.navigator_button)
        hbox.addWidget(self.detector_button)
        layout.addLayout(hbox)

    def handle_navigator(self):
        self.navigator.show()
    
    def handle_detector(self):
        self.detector.show()
    
    # def rec_audio(self):
    #     r = sr.Recognizer()  
    #     with sr.Microphone() as source:  
    #         print("Please wait. Calibrating microphone...")
    #         # listen for 5 seconds and create the ambient noise energy level  
    #         r.adjust_for_ambient_noise(source, duration=5)  
    #         print("Say something!")  
    #         audio = r.listen(source)

    #     text = r.recognize_google(audio)
    #     print("I think you said: " + text)
    #     return text

    # def listen(self):
    #     text = self.rec_audio()
    #     while True:
    #         if 'nav' in text.lower():
    #             self.navigator.show()
    #             break
    #         if 'det' in text.lower():
    #             self.detector.show()
    #             break

        

if __name__=='__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('the Light')
    menu = Menu()
    menu.show()
    # menu.listen()
    sys.exit(app.exec_())