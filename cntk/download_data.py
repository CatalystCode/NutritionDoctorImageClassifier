# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 11:53:31 2017

@author: tinzha
"""

from __future__ import print_function

import os
from azure.storage.blob import BlockBlobService
from math import floor
import random
import shutil

## CNTK download:
## https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Classification/ResNet

# ================================ CONFIGURATION =====================================
# Azure blob account information
ACCOUNT_NAME = "pinganhackfest2017"
ACCOUNT_KEY = "Hi7yuNxb67pBoSqwhHlnXRHnDcLyZmuVpbmc38vzA0j5HclHVIei66jIz+p7Qa9wobC8kUzBDFyI8LCe/842Ug=="
# input data in blob storage. each type of dish is stored in a separate container.
CONTAINER_NAMES = ['chow-mein1', 'kung-pao1', 'roujiamo1','burger1','sweet-sour1']

# training and testing split ratio
training_ratio = 0.8

# ================================ DEFINE FUNCTIONS =====================================
# check whether a directory exists. if not, create one
def assure_path_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

# obtain file list in a directory
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.jpg'), all_files))
    return data_files            

## prepare training dataset and test dataset with the split ratio
def get_training_and_testing_sets(split, file_list):
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

# ================================== DOWNLOAD DATA ====================================
## download data to a path: eg, chow-mein: DataSets/chow-main/*jpg
base_folder = os.path.dirname(os.path.abspath(__file__)) # Change me to store data elsewhere
dataset_folder = os.path.join(base_folder,"DataSets")
blob_service = BlockBlobService(account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)

if not os.path.exists(dataset_folder):
    os.mkdir(dataset_folder)
    print ('Downloading data from Azure Blob Storage ...')
    for container in CONTAINER_NAMES:
        generator = blob_service.list_blobs(container)
        file_num =0
        if not os.path.exists(os.path.join(dataset_folder, container)):
            os.mkdir(os.path.join(dataset_folder, container))
            file_path = os.path.join(dataset_folder, container)
            for blob in generator:
                file_num = file_num+1
                file_name = os.path.join(file_path, str(file_num)+'.jpg')
                try:
                    blob_service.get_blob_to_path(container, blob.name, file_name)
                    print('downloading successful')
                except:
                    print("something wrong happened when downloading the data %s"%blob.name)
   
# ============================ PREPARE TRAINING AND TESTING ================================           
## separate the datasets as training and testing then store the datasets in the following path
## filepath format under datasets: eg, chow-mein: DataSets/ChineseFood/Train/chow-main/*jpg

food_dir = os.path.join(dataset_folder, "ChineseFood")
assure_path_exists(food_dir)

dir_train = os.path.join(food_dir, 'Train')
dir_test = os.path.join(food_dir, 'Test')

for dir_data in [dir_train, dir_test]:
    if not os.path.exists(dir_data):
        os.mkdir(dir_data)  
        for dish_name in CONTAINER_NAMES:
            image_dir = os.path.join(dataset_folder, dish_name)       
            data_files = get_file_list_from_dir(image_dir)
            random.shuffle(data_files)
            training, testing = get_training_and_testing_sets(training_ratio,data_files)   
            
            sub_dir = os.path.join(dir_data, dish_name)          
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)     
                if dir_data.endswith('Train'):
                    dataset = training
                elif dir_data.endswith('Test'):
                    dataset = testing            
              
                for file_name in dataset:
                    if file_name.endswith('.jpg'):
                        shutil.copy(os.path.join(image_dir, file_name), sub_dir)
            else:
                print ("the dataset of %s has been created in %s" % (dish_name, dir_data))
               

# ====================== Create Output Folder =============================
output_path = os.path.join(base_folder, "Output/ChineseFood") 
os.makedirs(output_path, exist_ok = True)


            


