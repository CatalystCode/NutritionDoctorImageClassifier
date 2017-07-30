# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas_confusion import ConfusionMatrix
from cntk import load_model
from TransferLearning import *

# ================================ Load Model =====================================

# define base model location and characteristics
base_folder = os.path.dirname(os.path.abspath(__file__))
#base_model_file = os.path.join(base_folder,  "PretrainedModels", "ResNet18_ImageNet_CNTK.model")
#base_model_file = os.path.join(base_folder,  "PretrainedModels", "ResNet34_ImageNet_CNTK.model")
#base_model_file = os.path.join(base_folder,  "PretrainedModels", "ResNet50_ImageNet_CNTK.model")
base_model_file = os.path.join(base_folder,  "PretrainedModels", "ResNet50_ImageNet_Caffe.model")
#base_model_file = os.path.join(base_folder,  "PretrainedModels", "ResNet101_ImageNet_Caffe.model")
#base_model_file = os.path.join(base_folder, "PretrainedModels", "ResNet152_ImageNet_Caffe.model")

if base_model_file.endswith('CNTK.model'):
    # ResNetnn_ImageNet_CNTK.model settings
    feature_node_name = "features"
    last_hidden_node_name = "z.x"
elif base_model_file.endswith('Caffe.model'):
    # ResNetnn_ImageNet_caffe model settings
    feature_node_name = "data"
    last_hidden_node_name = "pool5"
else:
    print('Error: Unknown network definition')
    exit(1)

# ======================== Set Parameters and Data Location ===========================
image_height = 224
image_width = 224
num_channels = 3 

NODE_DUMP = False
NEPOCHS   = 100

# define data location and characteristics
train_image_folder = os.path.join(base_folder,  "DataSets", "ChineseFood", "Train")
test_image_folder = os.path.join(base_folder, "DataSets", "ChineseFood", "Test")
file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

# ================================ Define Functions ===================================

def create_map_file_from_folder(root_folder, class_mapping, include_unknown=False):
    map_file_name = os.path.join(root_folder, "map.txt")
    lines = []
    for class_id in range(0, len(class_mapping)):
        folder = os.path.join(root_folder, class_mapping[class_id])
        if os.path.exists(folder):
            for entry in os.listdir(folder):
                filename = os.path.join(folder, entry)
                if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                    lines.append("{0}\t{1}\n".format(filename, class_id))

    if include_unknown:
        for entry in os.listdir(root_folder):
            filename = os.path.join(root_folder, entry)
            if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                lines.append("{0}\t-1\n".format(filename))

    lines.sort()
    with open(map_file_name , 'w') as map_file:
        for line in lines:
            map_file.write(line)

    return map_file_name

def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    classes.sort()
    return np.asarray(classes)

def format_output_line_orig(img_name, true_class, probs, class_mapping, top_n=3):
    class_probs = np.column_stack((probs[0], class_mapping)).tolist()
    class_probs.sort(key=lambda x: float(x[0]), reverse=True)
    top_n = min(top_n, len(class_mapping)) if top_n > 0 else len(class_mapping)
    true_class_name = class_mapping[true_class] if true_class >= 0 else 'unknown'
    line = '[{"class": "%s", "predictions": {' % true_class_name
    for i in range(0, top_n):
        line = '%s"%s":%.3f, ' % (line, class_probs[i][1], float(class_probs[i][0]))
    line = '%s}, "image": "%s"}]\n' % (line[:-2], img_name.replace('\\', '/').rsplit('/', 1)[1])
    return line

def format_output_line(img_name, true_class, probs, class_mapping, top_n=3):
    class_probs = np.column_stack((probs[0], class_mapping)).tolist()
    class_probs.sort(key=lambda x: float(x[0]), reverse=True)
    top_n = min(top_n, len(class_mapping)) if top_n > 0 else len(class_mapping)
    true_class_name = class_mapping[true_class] if true_class >= 0 else 'unknown'
    line = '%s\t%s' % (img_name.replace('\\', '/').rsplit('/', 1)[1], true_class_name)
    for i in range(0, top_n):
        line = '%s\t%s\t%.3f' % (line, class_probs[i][1], float(class_probs[i][0]))
    line = '%s\n' % line
    return line

def train_and_eval(_base_model_file, class_mapping,  _train_image_folder, _test_image_folder, _results_file, _new_model_file, testing = False):
    # check for model and data existence
    if not (os.path.exists(_base_model_file) and os.path.exists(_train_image_folder) and os.path.exists(_test_image_folder)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)

    # get class mapping and map files from train and test image folder
    train_map_file = create_map_file_from_folder(_train_image_folder, class_mapping)
    test_map_file = create_map_file_from_folder(_test_image_folder, class_mapping, include_unknown=True)

    # train
    trained_model = train_model(_base_model_file, feature_node_name, last_hidden_node_name,
                                image_width, image_height, num_channels,
                                len(class_mapping), train_map_file, num_epochs=NEPOCHS, max_images = 20, freeze=True)

    if not testing:
        trained_model.save(_new_model_file)
        print("Stored trained model at %s" % _new_model_file)

    # evaluate test images
    total   = 0
    correct = 0
    with open(_results_file, 'w') as output_file:
        with open(test_map_file, "r") as input_file:
            for line in input_file:
                tokens = line.rstrip().split('\t')
                img_file = tokens[0]
                true_label = int(tokens[1])
                probs = eval_single_image(trained_model, img_file, image_width, image_height)
                predicted_label = np.argmax(probs)

                formatted_line = format_output_line(img_file, true_label, probs, class_mapping)
                output_file.write(formatted_line)
                total += 1
                if predicted_label == true_label:
                    correct += 1

    print("Done. Wrote output to %s" % _results_file)
    print ("{0} out of {1} predictions - accuracy = {2}.".format(correct, total, (float(correct) / total)))

def get_confusion_matrix(_results_file):
    df = pd.read_csv(results_file, sep='\t', header=None)
    true_lbls = df[1]
    pred_lbls = df[2]
    confusion_matrix = ConfusionMatrix(true_lbls, pred_lbls)
    confusion_matrix.plot()
    cm_file = _results_file.replace('.txt', '_cm.jpg')
    plt.savefig(cm_file)

    print()
    print(confusion_matrix)
    print()
    cm = confusion_matrix.to_dataframe()
    correct = 0
    for i in range(cm.shape[0]):
        correct += cm.iloc[i][i]
        recall = cm.iloc[i][i] * 100 / cm.sum(axis=0)[i]
        prec   = cm.iloc[i][i] * 100 / cm.sum(axis=1)[i]
        print ('Class %s recall = %.4f precision = %.4f' % (cm.columns[i], recall, prec))
    print('Overall accuracy = %.4f' % float(correct * 100 / sum(cm.sum(axis=0))))


if __name__ == '__main__':

    try_set_default_device(gpu(1)) # adjust the number based on your GPU machine

    # You can use the following to inspect the base model and determine the desired node names
    if NODE_DUMP:
        node_outputs = get_node_outputs(load_model(base_model_file))
        for out in node_outputs: print("{0} {1}".format(out.name, out.shape))
        exit(0)

    print('Base model = %s' % base_model_file)
    print('Last hidden layer = %s' % last_hidden_node_name)
    class_mapping = create_class_mapping_from_folder(train_image_folder)
    
    mapping_file = os.path.join(base_folder, "Output/ChineseFood",'mapping.dat')
    results_file = os.path.join(base_folder, "Output/ChineseFood", "predictions_chinesefood.txt")
    new_model_file = os.path.join(base_folder, "Output/ChineseFood", "TransferLearningChineseFood.model")

    save_mapping = pd.DataFrame(class_mapping)
    save_mapping.to_pickle(mapping_file)
    
    train_and_eval(base_model_file, class_mapping, train_image_folder, test_image_folder, results_file, new_model_file)
    get_confusion_matrix(results_file) # evaluate the built model using the testing dataset



