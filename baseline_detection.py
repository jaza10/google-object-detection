## import required libraries 
import matplotlib as mpl
mpl.use('Agg')

from imageai.Detection import ObjectDetection
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.utils.data_utils import GeneratorEnqueuer

import glob
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np 
import os
import pandas as pd 
import time

def zero_if_negative(x):
    """
    Some bounding boxes are negative. Those should be set to 0.
    """
    if x < 0:
        return 0.
    return x

def normalize_results(img, detection):
    # i.e. like  this
    h, w, _ = img.shape
    x1, y1, x2, y2 = detection['box_points']
    x1 /= w
    x2 /= w
    y1 /= h
    y2 /= h
    prob = detection['percentage_probability'] / 100.
    return detection['name'], prob, zero_if_negative(x1), zero_if_negative(x2), zero_if_negative(y1), zero_if_negative(y2)

def extract_img_ids(fname):
    """
    Retrieves all processed images from csv file.
    This way, even if something breaks, we can pick it up wherever we started.
    
    Parameters:
    --------------
    fname: str, results file
    
    Returns:
    --------------
    ids: list, image ids that have already been processed
    """
    ids = []
    with open(fname, 'r') as f:
        for line in f.readlines()[1:]:
            if line == '\n':
                continue
            if ',' in line:
                ids.append(line.split(',')[0])
    return ids

def detect_objects(detector_fn, img_path, results_fname, minimum_percentage_probability, translation_dict):
    """
    Applies detector_fn to every image in img_path. 
    Retrieves only .jpg files right now.
    Write results to .csv file for submission.
    
    Parameters:
    --------------
    detector_fn:                     python function for object detection
    img_path:                        str or os.path, file of images that should be processed
    results_fname:                   str or os.path, submission file
    minimum_percentage_probabilitiy: int, filtering percentage for detector_fn
    translation_dict:                dict, maps classes to google classes
    
    Returns:
    --------------
    None
    """ 
    # initiate results file or read already processed img ids
    if not os.path.exists(results_fname):
        with open(results_fname, 'a') as f:
            f.write('ImageId,PredictionString\n')
        img_collection = []
    else:
        img_collection = extract_img_ids(results_fname)
    
    images = glob.glob(os.path.join(img_path, '*.jpg'))
    n_images = len(images)
    for ix, img_fname in enumerate(images):
        # get image id from filename
        img_id = img_fname.split('/')[-1].split('.')[0]
        if img_id in img_collection:
            continue
        start_time = time.time()
        # read in data
        tmp_img = mpimg.imread(img_fname)
        # detect object per image
        tmp_detections = detector_fn(
            input_image=tmp_img, 
            input_type='array', 
            minimum_percentage_probability=minimum_percentage_probability)
        output_str = '{},'.format(img_id)
        
        # iterate over detections and transform
        for detection in tmp_detections:
            class_name, prob, x1, x2, y1, y2 = normalize_results(tmp_img, detection)
            if class_name not in translation_dict:  # class doesn't exist in google vocab
                continue
            tmp_str = "{} {} {} {} {} {} ".format(translation_dict[class_name], prob, x1, x2, y1, y2)
            output_str += tmp_str
            
        # write results to file
        with open(results_fname, 'a') as f:
            f.write(output_str.strip()) # remove last blank space
            f.write('\n')
        print ('Working on {} ({}/{}) - {:.2f} sec'.format(img_id, ix+1, n_images, time.time() - start_time))

def main():
    # setup folders
    image_path = "../kaggle/challenge2018_test/"
    model_weight_path = "resnet50_coco_best_v2.0.1.h5"

    print ('Reading and transformin class information ...')
    # Read classes from RetinaNet. Prepared list taken from: https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb
    labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    retina_classes = pd.DataFrame(list(labels_to_names.items()), columns=['ID', 'Label'])

    # Read classes from Google
    google_classes = pd.read_csv("class-descriptions-boxable.csv", delimiter=";", names=['ID', 'Label'])
    google_classes['Label'] = google_classes['Label'].apply(str.lower)

    # Iterate over Retina_classes and check if they're in Google classes.
    # If so, take google ID in new class mappings 

    label_mappings = pd.DataFrame(columns=['ID', 'Label'])
    coco_to_google = {}

    for i in range(0, retina_classes.shape[0]):
        for j in range(0, google_classes.shape[0]):
            if(retina_classes["Label"][i] == google_classes["Label"][j]):
                label_mappings.loc[i] = google_classes.iloc[j]
                coco_to_google[retina_classes["Label"][i]] = google_classes.iloc[j]['ID']

    print ('Loading resnet model ...')
    model_weight_path = "resnet50_coco_best_v2.0.1.h5"

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_weight_path)
    detector.loadModel()

    print ('Starting object detection method ...')
    detect_objects(
        detector_fn=detector.detectObjectsFromImage, 
        img_path=image_path,
        results_fname='challenge_submission.csv',
        minimum_percentage_probability=50,
        translation_dict=coco_to_google)

if __name__ == '__main__':
    main()

