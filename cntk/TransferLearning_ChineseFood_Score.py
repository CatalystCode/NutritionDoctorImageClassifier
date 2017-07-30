# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os, glob
from cntk import load_model
from TransferLearning import *
from pandas_confusion import ConfusionMatrix
import pandas as pd
from matplotlib import pyplot as plt
import urllib.request

#============================= set up parameters ============================
image_height = 224 # image height
image_width = 224 # image width
num_channels = 3 # color chanel

img_URL = "https://pinganhackfest2017.blob.core.windows.net/roujiamo1/1.jpg"

#============================ download image ================================
base_folder = os.path.dirname(os.path.abspath(__file__))
score_image_folder = os.path.join(base_folder, "Output", "ChineseFood", "Score")
if not os.path.exists(score_image_folder):
    os.mkdir(score_image_folder)
urllib.request.urlretrieve(img_URL, os.path.join(score_image_folder, "image.jpg")) #the downloaded images are stored under "C:\\temp"

# define base model location and characteristics
trained_model_file = os.path.join(base_folder, "Output/ChineseFood", "TransferLearningChineseFood.model") # the trained model
mapping_file = pd.read_pickle(os.path.join(base_folder, "Output/ChineseFood", "mapping.dat")) # mapping the label number with class name

# ========================= score the new image ===============================  
trained_model = load_model(trained_model_file) # load the trained model
img_file = os.path.join(score_image_folder, 'image.jpg')
probs = eval_single_image(trained_model, img_file, image_width, image_height) 
predicted_label = np.argmax(probs)

class_mapping = mapping_file[0].tolist() # map the label number to the label name

print ('the image label is', class_mapping[predicted_label])
    



