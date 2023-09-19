"""
Created on Wed Nov  9 13:52:03 2022

@author: juliana
"""

# %% Imports

import os
import glob
from utils import create_new_folder, create_file, get_cluster_interval, dunn_fast, cluster_kmeans, cluster_agglomerative, cluster_hdbscan
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "9"


# %% Import data - EPILEPSIAE

folder_name = 'Dimensionality reduction csv - Test'

path_folders = os.path.join(os.getcwd(), folder_name,'pat*')

folders_list = glob.glob(path_folders)

gridsearch_csv = pd.read_csv('gridsearch.csv', ';', header=None)

new_folder_name = 'Clusters'

create_new_folder(new_folder_name)

filename = 'metrics_cluster.csv'
header = ['patient', 'clustering','sc','dunn_idx', 'density', 'begin', 'end']

new_path = os.path.join(os.getcwd(),new_folder_name)

writer, file = create_file(new_path, filename, header, file_type='csv')


for path in folders_list:
    
    patient_seizure = path.split('/')[-1]
    
    dim_reduction = pd.read_csv(path).rename(columns={'Unnamed: 0':'Datetime', '0':'Axis x', '1':'Axis y', '2':'Axis z', '0.1':'AE - AH'})
    
    data = dim_reduction[['Axis x', 'Axis y', 'Axis z']]
    
    predict_kmeans, sc_index_kmeans = cluster_kmeans(data)
    
    predict_agg, sc_index_agg = cluster_agglomerative(data)
    
    predict_hdbscan, sc_index_hdbscan = cluster_hdbscan(data)
    
   
    preictal_begin_idx_kmeans, preictal_end_idx_kmeans, preictal_density_kmeans = get_cluster_interval(predict_kmeans, 6)
    preictal_begin_idx_agg, preictal_end_idx_agg, preictal_density_agg = get_cluster_interval(predict_agg, 6)
    preictal_begin_idx_hdbscan, preictal_end_idx_hdbscan, preictal_density_hdbscan = get_cluster_interval(predict_hdbscan, 6)

    dunn_index_kmeans = dunn_fast(data, predict_kmeans)
    dunn_index_agg = dunn_fast(data, predict_agg)
    dunn_index_hdbscan = dunn_fast(data, predict_hdbscan)
    
    writer.writerow([patient_seizure, 'kmeans',sc_index_kmeans, dunn_index_kmeans, preictal_density_kmeans, preictal_begin_idx_kmeans, preictal_end_idx_kmeans])
    writer.writerow([patient_seizure, 'ah', sc_index_agg, dunn_index_agg, preictal_density_agg, preictal_begin_idx_agg, preictal_end_idx_agg])
    writer.writerow([patient_seizure, 'hdbscan', sc_index_hdbscan, dunn_index_hdbscan, preictal_density_hdbscan, preictal_begin_idx_hdbscan, preictal_end_idx_hdbscan])

    # ## %% Saving
    
    data = pd.concat([dim_reduction, pd.DataFrame(predict_kmeans, columns=['KMeans']), pd.DataFrame(predict_agg, columns=['AH']), pd.DataFrame(predict_hdbscan, columns=['HDBSCAN'])],axis=1)
    
    data.to_csv(os.path.join(new_path, patient_seizure))
    
    print('\n Patient :', patient_seizure, '\n')
    

file.close()