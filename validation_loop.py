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

import thread
from multiprocessing import Queue

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

NUM_CLASSES = 27
SAVE_DIR = './output/'
RESTORE_PATH = './restore_weights/'
matfn = 'color150.mat'

def get_arguments():
    parser = argparse.ArgumentParser(description="Indoor segmentation parser.")
    parser.add_argument("--img_path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_PATH,
                        help="checkpoint location")

    return parser.parse_args()

def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list

def decode_labels(mask, num_images=1, num_classes=150):
    label_colours = read_labelcolours(matfn)

    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label_colours[k]
      outputs[i] = np.array(img)
    return outputs

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def imageToText(preds):

    # print (preds)
    preds_modify = np.where(preds != 4, np.zeros(preds.shape,dtype=np.int), np.ones(preds.shape,dtype=np.int))
    parts_name = ['left', 'center', 'right']
    part_idx = [int(round(preds.shape[2]/3)),int(round(preds.shape[2]*2/3))]
    height_idx = int(round(preds.shape[1]*2/3))
    parts = []
    parts.append(preds_modify[:,height_idx:,:part_idx[0],:])
    parts.append(preds_modify[:,height_idx:, part_idx[0]:part_idx[1], :])
    parts.append(preds_modify[:,height_idx:, part_idx[1]:, :])

    def isobstacle(part):
        percentage_all = np.sum(part)/float(part.size)
        center = part[:,part.shape[1]/4:,part.shape[2]/4:part.shape[2]*3/4,:]
        percentage_center = np.sum(center)/float(center.size)
        print (percentage_all, percentage_center)
        if percentage_all > 0.9 and percentage_center>0.99:
            return 0
        else:
            return 1
    out = ''

    obstacle_count = 0
    blocked = [0,0,0]
    for i in range(len(parts)):
        obstacle = isobstacle(parts[i])
        if obstacle == 1:
            if obstacle_count >= 1:
                out += ' and '
            out += ('Watch your ' + parts_name[i]) if out is '' else parts_name[i]
            obstacle_count += 1
            blocked[i] = 1
    if obstacle_count is 3:
        out = 'Slow down! All directions blocked.'

    out = 'All clear!' if out is '' else out
    return blocked, out

def speak(out):
    import pyttsx
    engine = pyttsx.init()
    engine.setProperty('voice', 'english+f3')
    engine.setProperty('rate', 150)
    # engine.setProperty('rate', rate - 1)
    engine.say(out)
    engine.runAndWait()

# args = get_arguments()

img_paths = ['./1.png', './2.jpg']
restore_from = './restore_weights'

# Create network.
input_placeholder = tf.placeholder(tf.float32, (1, 256, 256, 3))
net = DeepLabResNetModel({'data': input_placeholder}, is_training=False, num_classes=NUM_CLASSES)

# Which variables to load.
restore_var = tf.global_variables()

# Predictions.
raw_output = net.layers['fc_out']
raw_output_up = tf.image.resize_bilinear(raw_output, [256, 256])
raw_output_up = tf.argmax(raw_output_up, dimension=3)
pred = tf.expand_dims(raw_output_up, dim=3)

# Set up TF session and initialize variables.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

sess.run(init)

# Load weights.
ckpt = tf.train.get_checkpoint_state(restore_from)

if ckpt and ckpt.model_checkpoint_path:
    loader = tf.train.Saver(var_list=restore_var)
    load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
    load(loader, sess, ckpt.model_checkpoint_path)
else:
    print('No checkpoint file found.')
    load_step = 0

def prediction_thread_function(pred, img, q):
    preds = sess.run(pred, feed_dict={input_placeholder: img})
    blocked, out = imageToText(preds)

    q.put((blocked, out, preds))

def preProcess(img):
    img = misc.imresize(img, [256,256])
    img = img[:,:,[2,1,0]].astype(np.float32)
    img -= IMG_MEAN
    img = np.expand_dims(img, axis=0)
    return img

# img = misc.imread(filename[0])
def navigation(img):
    img = preProcess(img)
    q = Queue()
    thread.start_new_thread(prediction_thread_function, (pred, img, q))
    # can do something else
    blocked, out, _ = q.get()
    thread.start_new_thread(speak, (out,))
    return blocked, out

# preds = sess.run(pred, feed_dict={input_placeholder: img})
# cap = skvideo.io.VideoCapture('v.mp4', frameSize=(1080,1920))
# # for img_path in img_paths:
# for i in range(frames_to_process):
#
#     # filename = img_path.split('/')[-1]
#     #
#     # if os.path.isfile(img_path):
#     #     print('successful load img: {0}'.format(img_path))
#     # else:
#     #     print('not found file: {0}'.format(img_path))
#     #     sys.exit(0)
#
#     # Convert RGB to BGR.
#     for p in range(20):
#         if cap.isOpened():
#             _, img = cap.read()
#         else:
#             raise EOFError('Video-end Reached.')
#
#     # img = misc.imread(filename), [256,256]
#     img1 = img
#     img = misc.imresize(img, [256,256])
#
#     img = img[:,:,[2,1,0]].astype(np.float32)
#
#     # img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
#     # img = np.cast(np.concatenate((img_b, img_g, img_r), axis=2), np.float32)
#     # img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
#     # Extract mean.
#     img -= IMG_MEAN
#     img = np.expand_dims(img, axis=0)
#
#     # Perform inference.
#     preds = sess.run(pred, feed_dict={input_placeholder: img})
#
#     out = imageToText(preds)
#     print(out)
#     thread.start_new_thread(speak, (out,))
#
#     plt.show(plt.imshow(img1[:,:,[2,1,0]]))
#
#     msk = decode_labels(preds, num_classes=NUM_CLASSES)
#     im = Image.fromarray(msk[0])
#     if not os.path.exists(SAVE_DIR):
#         os.makedirs(SAVE_DIR)
#     im.save(SAVE_DIR + 'frame_' + str(i) + '.jpg')
#
#     print('The output file has been saved to {0}'.format(SAVE_DIR + 'frame_' + str(i) + '.jpg'))
#
# cap.release()
