"""
Created on Thu Apr  7 10:54:15 2022

@author: julia
"""

import pandas as pd
import numpy as np
import os
import glob
import csv
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pickle import dump
from sklearn import cluster, metrics
from collections import Counter
from sklearn.metrics import pairwise
import hdbscan 

from numpy.random import seed
seed(1)


def get_smallest_cluster(labels):
     
    count = Counter(labels).most_common()[-1]
    
    cluster_label = count[0]
    
    n_samples = count[1]
    
    return cluster_label, n_samples

def get_cluster_interval(labels,window):
    
    # interval = [-720:-60]
    
    min_dur = 30 
    max_dur = 600
    
    possible_preictal = labels[-720:-60]
    
    cluster_label, n_samples = get_smallest_cluster(labels)
    
    preictal_samples = Counter(possible_preictal)[cluster_label]
    
    percentage_preictal_samples_cluster = preictal_samples/n_samples
    
    if preictal_samples > min_dur and preictal_samples < max_dur and percentage_preictal_samples_cluster > 0.85:
        
        filtered_label = pd.DataFrame(labels).rolling(window, min_periods=1).mean().round(0).to_numpy()
        
        filtered_label = [i[0] for i in filtered_label]
        
        possible_preictal_filtered = filtered_label[-720:-60]
        
        preictal_begin_idx = np.where(possible_preictal_filtered == cluster_label)[0][0]
        preictal_end_idx = np.where(possible_preictal_filtered == cluster_label)[0][-1]
        
        preictal_range = preictal_end_idx - preictal_begin_idx + 1
        
        preictal_density = Counter(possible_preictal_filtered[preictal_begin_idx:preictal_end_idx])[cluster_label]/preictal_range
        
        return preictal_begin_idx, preictal_end_idx, preictal_density
    
    else:
        return 0, 0, 0
        
def get_time_from_idx(datetime, idx_begin, idx_end):
    
    return datetime[idx_begin], datetime[idx_end]

def get_sorted_folders(home_dir = ""):

    if len(home_dir) == 0:
        home_dir = os.getcwd()
        lst_original = glob.glob(os.path.join(home_dir, 'Features', 'pat*'))
    else:
        lst_original = glob.glob(os.path.join(home_dir, 'pat*'))

    if ".DS_Store" in lst_original:
        lst_original.remove(".DS_Store")

    split_lst = [i.split('_')[3] for i in lst_original]
    lst = [int(x) for x in split_lst]
    lst_original.sort(key=dict(zip(lst_original, lst)).get)

    return lst_original

def get_patient_and_seizure_id(filename, data_type = 'features'):
    
    if data_type == 'features':

        filename = filename.split('/')[-1]
    
        patient = filename.split('_')[1]
    
        seizure = filename.split('_')[3]
        
        return patient, seizure
        
    else:
        
        filename = filename.split('/')[-1]
    
        patient = filename.split('_')[1]

        return patient

def get_files(folder_path, data_type = 'features'):

    seizure_information = np.load(os.path.join(
        folder_path, "all_seizure_information.pkl"), allow_pickle=True)
    
    if data_type == 'features':

        features = glob.glob(os.path.join(folder_path, 'pat*.npy'))
    
        split_features = [i.split('_')[-2] for i in features]
        features_number = [int(x) for x in split_features]
        features.sort(key=dict(zip(features, features_number)).get)
    
        datetimes = glob.glob(os.path.join(folder_path, 'feature*.npy'))
    
        split_datetimes = [i.split('_')[-1] for i in datetimes]
        split_datetimes = [i.split('.')[0] for i in split_datetimes]
        datetimes_number = [int(x) for x in split_datetimes]
        datetimes.sort(key=dict(zip(datetimes, datetimes_number)).get)

    else:
        
        features =  np.load(os.path.join(folder_path, 'all_eeg_dataset.pkl'), allow_pickle=True)
        datetimes = np.load(os.path.join(folder_path, 'all_datetimes.pkl'), allow_pickle=True)
        

    return seizure_information, features, datetimes

def preprocessing(data, datetimes):
    
    for i in range(len(data)):

        seizure = data[i]

        seizure_datetime = datetimes[i]

        seizure_datetime = np.array(
            list(map(timestamp_to_datetime, seizure_datetime[:, 0])))

        seizure, datetime = reduce_to_four_and_half_hours(
            seizure, seizure_datetime, 'raw')

        data[i] = seizure
        datetimes[i] = datetime
        
    return data, datetimes

def timestamp_to_datetime(date):

    d = int(date)
    t = date - d

    date = datetime.datetime.fromtimestamp(d) + datetime.timedelta(seconds=t)

    return date

def reduce_to_four_and_half_hours(seizure, datetime = None, data_type = 'features'):

    if data_type == 'features':
        
        seizure = seizure.iloc[-3240:]
        
    else:
        
        seizure = seizure[-1620:]
        datetime = datetime[-1620:]

    return seizure,datetime

def create_figure_umap_reduction_plotly(seizure_id, patient_id, seizure, label, datetime):

    folder_name = 'Dimensionality reduction graphs - 8902'
    
    create_new_folder(folder_name)
    
    path = os.path.join(os.getcwd(), folder_name)
    
    x = seizure[:, 0]
    y = seizure[:, 1]
    z = seizure[:, 2]

    time = [i.strftime("%Hh%M") for i in datetime] 
    c = np.arange(seizure.shape[0])
    
    array_size = len(time)
    
    time_ticks = [time[0], time[int(array_size/4)], time[int(array_size/2)], time[int(3*array_size/4)],time[-1]]
    c_ticks = [c[0], c[int(array_size/4)], c[int(array_size/2)], c[int(3*array_size/4)], c[-1]]

    label_filtered = pd.DataFrame(label).rolling(9, min_periods=1, center=True).mean().round(0).to_numpy()
    label_filtered = [int(i[0]) for i in label_filtered]


    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}], [{'type': 'xy', 'colspan': 2}, None]],
                        subplot_titles=('Hierarchical Clustering', 'Dimensionality Reduction - AE', 'Classes over time'),
                        shared_yaxes=False, shared_xaxes=True, horizontal_spacing=0.05, vertical_spacing=0.1)

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, 
                               marker=dict(size=3, color=c, colorscale='Viridis', 
                               showscale=True, colorbar=dict(len=0.48,  x=1, y=0.85,
                                                             tickvals=c_ticks, ticktext=time_ticks, 
                                                             ticks='outside', tickmode='array', title="Seizure onset")), 
                               mode='markers'),row=1, col=2)
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, 
                               marker=dict(size=3, color=label, colorscale='Viridis'), 
                               mode='markers'),row=1, col=1)
    
    fig.add_trace(go.Scatter(x=datetime, y=label_filtered, mode='lines+markers', name='clustering solution',
                              line=dict(dash='dash', color='rgb(189,189,189)')), row=2, col=1)

    fig.add_trace(go.Scatter(x=datetime, y=label_filtered, mode='markers',
                              marker=dict(color=label, colorscale='Viridis')), row=2, col=1)
    
    fig.update_xaxes(tickformat = '%Hh%M', row=2, col=1)
    

    fig.update_layout(showlegend=False, title_text='Patient: ' + str(patient_id) +
                      ' Seizure: '+str(seizure_id), margin=dict(l=80, r=10, t=70, b=20))

    fig.show()

    fig.write_html(os.path.join(path, 'pat_' + str(patient_id) +'_seizure_' + str(seizure_id) + '.html'))
    
def save_dim_reduction(seizure, datetime, label, seizure_id, patient_id):
    
    folder_name = 'Dimensionality reduction csv - 8902'
    
    create_new_folder(folder_name)
    
    path = os.path.join(os.getcwd(), folder_name,'pat_' + str(patient_id) +'_seizure_' + str(seizure_id) + '.csv')
    
    seizure = pd.DataFrame(seizure)
    
    label = pd.DataFrame(label)
    
    seizure = pd.concat([seizure,label],axis=1).set_index(datetime)
    
    seizure.to_csv(path)
    
def save_model(model, filename, path):

    filename = path + '/' + filename

    dump(model, open(filename, 'wb'))

def cluster_kmeans(data):

    model = cluster.KMeans(n_clusters=2, random_state=42)

    predict = model.fit_predict(data)

    sc_index = metrics.silhouette_score(data, predict)
    
    return predict, sc_index
   
def cluster_agglomerative(data):

    model = cluster.AgglomerativeClustering(n_clusters=2)

    predict = model.fit_predict(data)
    
    sc_index = metrics.silhouette_score(data, predict)
    
    return predict, sc_index

def cluster_hdbscan(data):
    
    min_cluster_size = 30  # minimum of 5 min of preictal (considering that each 5 seconds we have a sample)
    
    model = hdbscan.HDBSCAN(min_samples=6, min_cluster_size=min_cluster_size).fit(data)
    
    predict = model.labels_
    
    try:
        sc_index = metrics.silhouette_score(data, predict)
        
    except Exception as e:
        sc_index = '1 class'
        print(e)
    
    return predict, sc_index
   
def create_file(path, filename, header, file_type='txt'):
    
    if file_type == 'csv':
    
        file = open(os.path.join(path, filename), 'a')
        writer = csv.writer(file)
        writer.writerow(header)

        return writer, file
    
    else:
        
        file = open(os.path.join(path, filename), 'a')
        
        return file

def create_new_folder(folder_name, path = ''):
    
    if len(path) == 0:
        
        path = os.getcwd()

    path_old = os.path.join(path, folder_name)

    os.makedirs(path_old, exist_ok=True) 
    
def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = pairwise.euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    
    return di
        
def target_distribution(q):
    weight = q ** 2 / (q**2).sum(0)
    return (weight.T / weight.sum(1)).T




    

