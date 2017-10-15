# The Light

## Inspiration
Today(15 Oct) is the White Cane Safety Day which is set to celebrate the achievements of people who are blind or visually impaired.

There are over 284 million people who are experiencing vision loss worldwidely, where over 39 million people are totaly blind. The loss of visual capabilities limited them from moving independently. Instead, they need the help of other human or assistance of guide dogs. However these assistance induces high cost.

In order to contribute in building an inclusive society, we aim to help them through applying our specialization. We feel it's a good idea to use Artificial Intelligence to help them move around and navigate both indoors and outdoors. The technology can be easily exploited on portable devices such as mobile, which can be cheap and easy to use.

## What it Does
Our software have two functions.

The first and the most important one is that it helps blind people navigate and move around both indoors and outdoors by avoiding obstacles.

Secondly, it can detect and classify the objects surrounding them. 

## How we Built it
### Navigation Model
We used an image semantic augmentation model to detect the obstacles from a video/camera. 

Image Semantic Augmentation is a computer vision task that gives each pixel a label. The model we used is a state of the art model called DeepLab. The model is pre-trained and there are 27 classes. In order to improve accuracy and reduce redundancy, we modified it to a binary class detection: clear or not clear. 'Clear' means there is no obstacle and is walkable for blind people. 

In order to give suggestions, we segment the input video frame into three parts: left, center and right. We write an algorithm to check whether each part of the frame is clear. At last, our model will output one label for each of the three parts.

We are using Tensorflow as our deep learning framework.
### Object detection Model
As for object detection, we used Single Shot MultiBox Detector(SSD) model, which is a deep neural network that outputs the position and the label of objects.

We will tell the user the object name and its position for each predicted object in audio.
### Audio Generation
We generate a text(suggestion) from the model prediction first then convert the text into audio format.
### Speech Recognition
In order to make easier to control for blind people, we use the CMU Sphinx speech recognition engine to control the function selection and control in our software.
### GUI Design
As for the GUI design, we did not pay too much attention since our main focus is the Machine Learning algorithms. So here we just used the PyQt4 to develop a very simple GUI to demonstrate the basic functions of our software.

## Challenges We Ran Into
### Real Time Processing
The original model takes 40 seconds to run a single image(video snapshot), which makes it impossible to do real time obstacle detection. 

We added a data augmentation process in the model to reduce the running time. That is, we downsample the input image first before we run the prediction model.

Secondly we improved the software application structure. Instead of doing one snapshot a time, we modify the application to do sequential images prediction, so that the tensorflow session only needs to be loaded and initialized once.
### Video Processing
The original model is for processing images, while our input is real time camera video. 

Through researching and careful study, we figured out the necessary video processing methods.
### Audio Generation
It's a bit hard to find a clear and nice voice. We considered to record human voices. However we decided to continue using the audio synthesizer since it's more flexible.

### Integration into Application GUI
In order to parallel the multiple tasks, including video processing, inference and GUI controlling, we use a multi-threads design. However, synchronizing the threads is indeed a challenging task, especially handling the delay of the inference.


## Accomplishments that We're Proud of
### High Accuracy
We are able to achieve very high accuracy for various scenes. 

The model we used is DeepLab which is the state-of-the-art technology. DeepLab produces almost the current best result and the original paper is published on TPAMI which is one of the best Journal in Artificial Intelligence.
### Indoor Extension
The original task is only for outdoors scene detection for self-driving cars. Now we are able to extend the model to indoors navigation.
### Real Time
Our software is able to produce live time audio suggestions for camera video with delays less then one second.
### Easy to Use
Our software is user friendly for blind people since it uses audio to control and output audio suggestions. It does not require any external assistance.
### Inclusive Design
We are pround that we are able make some contributions to build the inclusive society and help others.


## What We Learned
### Software Design
We learned how to design and build a software from scrach, including the architecture design and the interaction of each components and objects.
### Problem Solving Skills
We enhanced our ability to solve problems. We learnt how to research, gather information and develop solutions efficiently.
### Project Management
We learnt how to manage a project. We started with brain storming, each of us propose some ideas then followed by discussion. Next we finalized our idea and come up with schedules and division of work.
### Collabration Skills and Communication Skills
The most important soft skill we learnt is how to collaberate with teammates and communicate effectively.


## What's Next
### Build on Portable Devices
Currently, the demo is on our laptop. Building it on portable devices will be our next step. It can be built in wearable devices that specially designed for blind people or on mobile devices, such as Android and IOS.
### Integerate with GPS
Our application will be more complete if we can integrate it with GPS which provides a route planner.
### Add More Functionality
We plan to add more funcionalities to our application to make it the only one application that the blind people needs. 

Functionalities that we are considering are currency recoginition, face recognition(recoginize friends and others' facial expression), text to audio tool, scene recogition and video summarization. 
