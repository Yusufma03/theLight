import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time

def detect(img):
    start = time.time()

    # This is needed to display the images.
    # matplotlib inline

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")

    from utils import label_map_util

    from utils import visualization_utils as vis_util

    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    # MODEL_FILE = MODEL_NAME + '.tar.gz'
    # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.

    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    # opener = urllib.request.URLopener()
    # opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    # tar_file = tarfile.open(MODEL_FILE)
    # for file in tar_file.getmembers():
    #     file_name = os.path.basename(file.name)
    #     if 'frozen_inference_graph.pb' in file_name:
    #         tar_file.extract(file, os.getcwd())

    load_model_start = time.time()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    load_model_end = time.time()

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    IMG_NAME = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
    TEST_IMAGE_PATHS = []
    # for i in range(len(IMG_NAME)):
    #     TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i + 1)))

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

    def calculate_cls_num(new_boxes, new_classes, class_names, left_bord, right_bord):
        cls_num = np.zeros(len(class_names), dtype=np.int64)

        for i in range(len(new_boxes)):
            ymin, xmin, ymax, xmax = new_boxes[i]
            im_width = 256
            im_height = 256
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            mid = int((left + right) / 2)
            if mid > left_bord and mid < right_bord:
                cls_num[class_names.index(new_classes[i])] += 1

        return cls_num

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Initiate the session
            init = tf.global_variables_initializer()
            sess.run(init)

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            counter = 0

            inf_times = []
            TEST_IMAGE_PATHS.append(img)
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)

                # resize image into 256x256
                image.resize((256, 256))

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                inf_start = time.time()
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
                inf_end = time.time()
                # Visualization of the results of a detection.
                box_category = vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                            np.squeeze(boxes),
                                                            np.squeeze(classes).astype(np.int32),
                                                            np.squeeze(scores),
                                                            category_index,
                                                            use_normalized_coordinates=True,
                                                            line_thickness=8)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
                plt.imsave(IMG_NAME[counter] + '_result.jpg', image_np)
                counter += 1
                inf_times.append(inf_end - inf_start)

                new_boxes = box_category.keys()
                new_classes = box_category.values()

                class_names = []
                for i in range(len(category_index.values())):
                    class_names.append(category_index.values()[i]['name'])

                cls_num_left = calculate_cls_num(new_boxes, new_classes, class_names, 0, 85)
                cls_num_mid = calculate_cls_num(new_boxes, new_classes, class_names, 86, 170)
                cls_num_right = calculate_cls_num(new_boxes, new_classes, class_names, 171, 255)

                ret_left = []
                ret_mid = []
                ret_right = []

                for i in range(len(cls_num_left)):
                    if cls_num_left[i] > 0:
                        # There are #cls_num_left[i] of class_names[i] in left-hand side
                        print('There are {} {}s in left-hand side.'.format(cls_num_left[i], class_names[i]))
                        ret_left.append([cls_num_left[i], class_names[i]])

                for i in range(len(cls_num_mid)):
                    if cls_num_mid[i] > 0:
                        # There are #cls_num_left[i] of class_names[i] in the middle
                        print('There are {} {}s in the middle.'.format(cls_num_mid[i], class_names[i]))
                        ret_mid.append([cls_num_mid[i], class_names[i]])

                for i in range(len(cls_num_right)):
                    if cls_num_right[i] > 0:
                        # There are #cls_num_left[i] of class_names[i] in right-hand side
                        print('There are {} {}s in right-hand side.'.format(cls_num_right[i], class_names[i]))
                        ret_right.append([cls_num_right[i], class_names[i]])
    
    return ret_left, ret_mid, ret_right

    end = time.time()
