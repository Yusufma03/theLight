from navigator import *
from PyQt4 import QtGui, QtCore
import speech_recognition as sr

class Menu(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.logo = QtGui.QLabel()
        self.logo.setPixmap(QtGui.QPixmap("./logo_resized.png"))
        self.navigator = Navigator()
        self.navigator_button = QtGui.QPushButton('Navigator', self)
        self.navigator_button.setText('Navigator')
        self.navigator_button.clicked.connect(self.handle_navigator)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.logo)
        layout.addWidget(self.navigator_button)
        

    def handle_navigator(self):
        self.navigator.show()

    def rec_audio(self):
        r = sr.Recognizer()  
        with sr.Microphone() as source:  
            print("Please wait. Calibrating microphone...")
            # listen for 5 seconds and create the ambient noise energy level  
            r.adjust_for_ambient_noise(source, duration=5)  
            print("Say something!")  
            audio = r.listen(source)

        text = r.recognize_google(audio)
        print("I think you said: " + text)
        return text

    def listen(self):
        text = self.rec_audio()
        while True:
            if 'nav' in text.lower():
                self.navigator.show()
                break
            else:
                text = self.rec_audio()

        

if __name__=='__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('the Light')
    menu = Menu()
    menu.show()
    menu.listen()
    sys.exit(app.exec_())