import matplotlib.pyplot as plt
import random
import sys
import os
from PIL import Image
from glob import glob

import numpy as np



def get_styles(data_dir):
    folders = os.listdir(data_dir)
    styles={}
    for f in folders:
        styles[f]=len(os.listdir(os.path.join(data_dir,f)))
    return styles        

def get_data(data_dir):
    data = glob(data_dir+"/*/*", recursive=True)
    return data

def get_labels(data):
    labels = []
    for i in range(len(data)):
        start_idx = data[i].find("art\\")+4
        modded = data[i][start_idx:]
        stop_idx = modded.find("\\")
        labels.append(modded[:stop_idx])
    return labels

def get_image(image_path, mode):
    image = Image.open(image_path)
    return np.array(image.convert(mode))
# ty gangogh

def get_train_and_test_sets(styles,split_percentage):
    train_set_image_names = {}
    train_set_image_names = {}
    for key,value in styles.items():
        number_per_style = range(value)
        random.shuffle(list(number_per_style))
        train_set_image_names[key] = number_per_style[:value//split_percentage]
        train_set_image_names[key] = number_per_style[value//split_percentage:]
    return train_set_image_names, train_set_image_names