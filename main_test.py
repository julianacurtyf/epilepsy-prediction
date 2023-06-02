#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:52:03 2022

@author: juliana
"""

# %% Imports

import os
import numpy as np
from keras.models import load_model
from utils import *
from tensorflow.keras.models import Model
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "9"


# %% Import data - EPILEPSIAE

path_to_raw_data = '/mnt/6a3bf9e0-7462-43d9-b6ae-3aa1a8be2f6a/fabioacl/Fabio/Fabio_Task_3/Datasets/'

folders_list = get_sorted_folders(path_to_raw_data)

gridsearch_csv = pd.read_csv('gridsearch.csv', ';', header=None)

folder_name = 'AE models - 8902'

create_new_folder(folder_name)

filename = 'metrics_test8902.csv'
header = ['patient','seizure', 'dunn_idx', 'density', 'begin', 'end']

writer, file = create_file(os.path.join(os.getcwd(),folder_name), filename, header, file_type='csv')


for j in range(len(folders_list)):
    
    seizure_information, data, datetimes = get_files(folders_list[j], 'raw')
    
    patient_id = get_patient_and_seizure_id(folders_list[j], 'raw')
    
    data, datetime = preprocessing(data, datetimes)
    
    numberSeizures = len(data)
    
    idx = eval(gridsearch_csv[1][j])
    
    train_set = np.concatenate((data[idx[0]],data[idx[1]]))
    target_train = np.concatenate((datetime[idx[0]],datetime[idx[1]]))
    
    mean_train_set = np.mean(train_set)
    std_train_set = np.std(train_set)
    
    for i in range(numberSeizures):
        
        test_set = data[i]
        
        test_target = datetime[i]
        
        test_set = (test_set - mean_train_set)/std_train_set
    
        
        # %% Testing
        
        seizure_number = i
        
        deepClusteringAutoencoder = load_model(os.path.join(os.getcwd(), folder_name, 'pat_8902'))
        
        test_reconstruction, test_pred = deepClusteringAutoencoder.predict(test_set)
        test_pred = np.argmax(test_pred, axis=1)
        
        latent = deepClusteringAutoencoder.get_layer('latent').output
        
        latentModel = Model(deepClusteringAutoencoder.input, latent)
        
        reduced_df = latentModel.predict(test_set)
        
        preictal_begin_idx, preictal_end_idx, preictal_density = get_cluster_interval(test_pred, 6)
        
        preictal_begin, preictal_end = get_time_from_idx(datetime[seizure_number], preictal_begin_idx, preictal_end_idx)
        
        dunn_index = dunn_fast(reduced_df, test_pred)
        
        writer.writerow([patient_id, seizure_number, dunn_index, preictal_density, preictal_begin_idx, preictal_end_idx])
    
        # # %% Plotting
        
        create_figure_umap_reduction_plotly(str(seizure_number), patient_id, reduced_df, test_pred, test_target)
        
        # ## %% Saving
        
        save_dim_reduction(reduced_df, test_target, test_pred, str(seizure_number),patient_id)
        
        print('\n Patient :', patient_id, '\n Seizure: ', seizure_number)
        

file.close()
# %% Plotting

# plot_3d(reduced_df, 'AE')

# patient_id = 8902

# create_figure_umap_reduction_plotly(str(seizure_number), patient_id, reduced_df, train_pred, datetime[seizure_number])

# %% Saving

# save_dim_reduction(reduced_df, datetime[seizure_number], train_pred, str(seizure_number),patient_id)

# folder_name = 'AE models'

# create_new_folder(folder_name)

# deepClusteringAutoencoder.save(os.path.join(os.getcwd(), folder_name, 'pat_' + str(patient_id) +'_seizure_' + str(seizure_number)))

