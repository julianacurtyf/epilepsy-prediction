#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:54:15 2022

@author: julia
"""

import pandas as pd
import numpy as np
import umap
import os
import glob
import csv
import datetime
from datetime import timedelta
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from pickle import dump, load
from sklearn import cluster, metrics
import keras.backend as K
import itertools
import scipy
from scipy.signal import periodogram
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from statsmodels.tsa import stattools
import pywt
from collections import Counter
from sklearn.metrics import pairwise
import hdbscan 

from numpy.random import seed
seed(1)


# %% Utils Juliana

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
    

def get_umap_reduction(train_set, validation_set, n_dim_umap):
    
    train_set = train_set.reshape([train_set.shape[0], train_set.shape[1]*train_set.shape[2]])
    validation_set = validation_set.reshape([validation_set.shape[0], validation_set.shape[1]*validation_set.shape[2]])
    
    umap_red = umap.UMAP(n_neighbors=10, n_components=n_dim_umap, min_dist=0.1, metric='correlation',random_state=42).fit_transform(train_set)
    
    umap_labels =  AgglomerativeClustering(
                n_clusters=2, affinity='euclidean', linkage='complete').fit(umap_red).labels_
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(umap_red[:,0],umap_red[:,1],umap_red[:,2], c=np.arange(len(umap_red[:,0])), marker='.')


def get_pca_reduction(train_set, validation_set, n_dim_pca):
    
    train_set = train_set.reshape([train_set.shape[0], train_set.shape[1]*train_set.shape[2]])
    validation_set = validation_set.reshape([validation_set.shape[0], validation_set.shape[1]*validation_set.shape[2]])

    pca_red = PCA(n_components=n_dim_pca).fit_transform(train_set)
    
    pca_labels =  AgglomerativeClustering(
                n_clusters=2, affinity='euclidean', linkage='complete').fit(pca_red).labels_
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(pca_red[:,0],pca_red[:,1],pca_red[:,2], c=np.arange(len(pca_red[:,0])), marker='.')


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


def standardize_data(df, data_type='features'):
    
    scaler = StandardScaler()
    
    if data_type == 'features':
    
        df_scaled = scaler.fit_transform(df)
    
        df_scaled = pd.DataFrame(df_scaled).set_index(df.index)
        
    else:
        
        df = scaler.fit_transform(df.reshape(-1, df.shape[-1])).reshape(df.shape)


    return df


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


def feat_selection_redundancy(features, threshold):

    corr_matrix = features.corr().values

    list_to_drop = []

    columns = features.columns

    for i in range(len(corr_matrix)):
        for j in range(i, len(corr_matrix)):
            if abs(corr_matrix[i, j]) >= threshold and i != j and j not in list_to_drop:
                list_to_drop.append(columns[j])

    features.drop(list_to_drop, axis=1, inplace=True)

    return features


def preprocessing(data,datetimes):
    
    for i in range(len(data)):

        seizure = data[i]

        seizure_datetime = datetimes[i]

        seizure_datetime = np.array(
            list(map(timestamp_to_datetime, seizure_datetime[:, 0])))

        seizure, datetime = reduce_to_four_and_half_hours(
            seizure, seizure_datetime, 'raw')
        
        # seizure = standardize_data(seizure, 'raw')

        data[i] = seizure
        datetimes[i] = datetime
        
    return data, datetimes


def feat_selection_variance(features, threshold):

    var_matrix = features.var().values

    list_to_drop = []

    for i in range(len(var_matrix)):
        if abs(var_matrix[i]) <= threshold:
            list_to_drop.append(i)

    features.drop(list_to_drop, axis=1, inplace=True)

    return features


def timestamp_to_datetime(date):

    d = int(date)
    t = date - d

    date = datetime.datetime.fromtimestamp(d) + datetime.timedelta(seconds=t)

    return date


def remove_SPH_from_signal(seizure_data, seizure_datetime, seizure_information):

    seizure_onset = timestamp_to_datetime(seizure_information[0])

    sph_datetime = seizure_onset - timedelta(minutes=10)

    final_index = 0

    # we search for all datetimes and compare it with the onset-SPH time
    for i in range(len(seizure_datetime)-1, 1, -1):

        current_datetime = timestamp_to_datetime(seizure_datetime[i])

        # when we reach the onset-SPH time, we stop
        if not(current_datetime > sph_datetime):
            final_index = i
            break

    # we keep the elements from the beginning until the SPH moment
    seizure_datetime = seizure_datetime[0:final_index]
    seizure_data = seizure_data[0:final_index, :, :]

    return seizure_data, seizure_datetime


def drop_nan(file):

    file_df = file.dropna(axis=0)

    return file_df


def array_to_df(file, index):
    
    file = np.reshape(file, (file.shape[0], file.shape[1]*file.shape[2]))

    file_df = pd.DataFrame(file, index=index)
        
    return file_df


def reduce_to_four_and_half_hours(seizure, datetime = None, data_type = 'features'):

    if data_type == 'features':
        
        seizure = seizure.iloc[-3240:]
        
    else:
        
        seizure = seizure[-1620:]
        datetime = datetime[-1620:]

    return seizure,datetime


def plot_3d(data, name):

    fig1 = plt.figure(1)
    ax = Axes3D(plt.figure(1), title='Seizure')
    p1 = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=range(data.shape[0]),marker='.')
    c = range(data.shape[0])
    cbar = fig1.colorbar(p1, ticks=[0, c[int(data.shape[0]/2)], c[-1]],shrink=0.8)
    a = round(c[-1]/6, 1)
    cbar.ax.set_yticklabels([str(a), str(a/2), '0'])
    ax.set_title(name + ' 3D', fontsize=16, fontstyle='oblique', color='k')


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


def get_model(filename, path):

    filename = path + filename

    model = load(open(filename, 'rb'))

    return model


def cluster_kmeans(data):

    model = cluster.KMeans(n_clusters=2, random_state=42)

    predict = model.fit_predict(data)

    sc_index = metrics.silhouette_score(data, predict)
    
    # save_model(model, patient_id + '_kmeans_2.sav', path)
    
    return predict, sc_index
   
   
def cluster_agglomerative(data):

    model = cluster.AgglomerativeClustering(n_clusters=2)

    predict = model.fit_predict(data)
    
    sc_index = metrics.silhouette_score(data, predict)
    
    # save_model(model, 'pat_' + patient_id + '_agglomerative_2.sav', path)
    
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
    
    # save_model(model, 'pat_' + patient_id + '_agglomerative_2.sav', path)
    
    return predict, sc_index
   
def create_file(path, filename, header, file_type='txt'):
    
    if file_type == 'csv':
    
        file = open(os.path.join(path, filename), 'a')
        writer = csv.writer(file)
        #writer.writerow(header)

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
        
    
# %% Utils - Fábio 


def correctClusterProbabilities(predProbs,twoHoursIndex):
    predictedLabels = np.argmax(predProbs,axis=1)
    interictalProbs = predProbs[0:twoHoursIndex]
    interictalLabels = predictedLabels[0:twoHoursIndex]
    preictalProbs = predProbs[twoHoursIndex:]
    preictalLabels = predictedLabels[twoHoursIndex:]
    
    # NAO DEIXAR CAIR AMOSTRAS DE PREICTAL NO PRIMEIRO INTERVALO

    return None
  


def fromSignalsToSpectrograms(data,fs=256,noverlap=128,window='hanning',removeFiltered = True):
    
    newData = []
    
    for sample in data:
        newSample = []
        numberChannels = sample.shape[1]
        for channelIdx in range(numberChannels):
            _,_,s = scipy.signal.spectrogram(sample[:,channelIdx],fs=fs,noverlap=noverlap,window=window)
            s = s[1:91,:]
            s = s.T
            newSample.append(s)
        newSample = np.array(newSample)
        newData.append(newSample)
    newData = np.array(newData)
    
    return newData
            

def eucl(x, y):
    """
    Euclidean distance between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    d = np.sqrt(np.sum(np.square(x - y), axis=0))
    return np.sum(d)


def cid(x, y):
    """
    Complexity-Invariant Distance (CID) between two multivariate time series given as arrays of shape (timesteps, dim)
    Reference: Batista, Wang & Keogh (2011). A Complexity-Invariant Distance Measure for Time Series. https://doi.org/10.1137/1.9781611972818.60
    """
    assert(len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    ce_x = np.sqrt(np.sum(np.square(np.diff(x, axis=0)), axis=0) + 1e-9)
    ce_y = np.sqrt(np.sum(np.square(np.diff(y, axis=0)), axis=0) + 1e-9)
    d = np.sqrt(np.sum(np.square(x - y), axis=0)) * np.divide(np.maximum(ce_x, ce_y), np.minimum(ce_x, ce_y))
    return np.sum(d)


def cor(x, y):
    """
    Correlation-based distance (COR) between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    scaler = TimeSeriesScalerMeanVariance()
    x_norm = scaler.fit_transform(x)
    y_norm = scaler.fit_transform(y)
    pcc = np.mean(x_norm * y_norm)  # Pearson correlation coefficients
    d = np.sqrt(2.0 * (1.0 - pcc + 1e-9))  # correlation-based similarities
    return np.sum(d)


def acf(x, y):
    """
    Autocorrelation-based distance (ACF) between two multivariate time series given as arrays of shape (timesteps, dim)
    Computes a linearly weighted euclidean distance between the autocorrelation coefficients of the input time series.
    Reference: Galeano & Pena (2000). Multivariate Analysis in Vector Time Series.
    """
    assert (len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    x_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, x)
    y_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, y)
    weights = np.linspace(1.0, 0.0, x.shape[0])
    d = np.sqrt(np.sum(np.expand_dims(weights, axis=1) * np.square(x_acf - y_acf), axis=0))
    return np.sum(d)

def dist(x1, x2, distMetric):
    """
    Compute distance between two multivariate time series using chosen distance metric
    # Arguments
        x1: first input (np array)
        x2: second input (np array)
    # Return
        distance
    """
    if distMetric == 'eucl':
        return eucl(x1, x2)
    elif distMetric == 'cid':
        return cid(x1, x2)
    elif distMetric == 'cor':
        return cor(x1, x2)
    elif distMetric == 'acf':
        return acf(x1, x2)
    else:
        raise ValueError('Available distances are eucl, cid, cor and acf!')

def target_distribution(q):
    weight = q ** 2 / (q**2).sum(0)
    return (weight.T / weight.sum(1)).T

def rrmse(yTrue,yPred):
    yDiff = yTrue-yPred
    rmsYDiff = K.sqrt(K.mean(K.square(yDiff)))
    rmsYTrue = K.sqrt(K.mean(K.square(yTrue)))
    return rmsYDiff/rmsYTrue

'''Listdir method returning full path filenames. Imported from 
https://stackoverflow.com/questions/120656/directory-tree-listing-in-python'''
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


'''Get interictal and preictal data. Seizure Prediction Horizon (SPH) represents the interval that
the patient has to prepare himself/herself to the seizure, i.e., if a SPH of 10 minutes is used,
the seizure only happens at least 10 minutes after the seizure alarm. Seizure Occurence Period (SOP) 
represents the interval when the seizure occurs. For predicting the seizure, a preictal interval 
equals to the SOP is used. Therefore, it is expected that the seizure occurs after the sph and
inside the following sop interval. For example, if a SOP interval of 40 minutes is used 
it is expected that the seizure occurs inside the 40 minutes after the SPH.'''
def getInterictalAndPreictalData(allSeizuresWindowsData,allSeizuresWindowsDatetimes,usedSeizureDatetimes,
                                 sopTime,sphTime):
    sopDatetime = pd.Timedelta(minutes=sopTime)
    sphDatetime = pd.Timedelta(minutes=sphTime)
    allWindowsClasses = []
    allWindowsDataWithoutSph = []
    allWindowsDatetimesWithoutSph = []
    for seizureIndex,seizure in enumerate(allSeizuresWindowsData):
        # Append arrays to separate each seizure segments
        allWindowsDataWithoutSph.append([])
        allWindowsClasses.append([])
        allWindowsDatetimesWithoutSph.append([])
        # Get segment datetimes
        seizureDatetimes = allSeizuresWindowsDatetimes[seizureIndex]
        # Get interictal plus tolerance begin, seizure begin and seizure end datetimes
        actualSeizureDatetimes = usedSeizureDatetimes[seizureIndex]
        # Get beginning seizure's datetime
        beginSeizureDatetime = pd.to_datetime(actualSeizureDatetimes[1])
        # Get beginning and end preictal datetime
        endPreictalDatetime = beginSeizureDatetime - sphDatetime
        beginPreictalDatetime = endPreictalDatetime - sopDatetime
        for windowIndex,windowData in enumerate(seizure):
            windowDatetimes = seizureDatetimes[windowIndex]
            beginWindowDatetime = windowDatetimes[0]
            endWindowDatetime = windowDatetimes[-1]
            # We only use data before the sph interval
            if beginWindowDatetime<endPreictalDatetime:
                allWindowsDataWithoutSph[seizureIndex].append(windowData)
                allWindowsDatetimesWithoutSph[seizureIndex].append(windowDatetimes)
                # If the window data finishes inside preictal period it is considered as preictal
                # otherwise it is considered interictal
                if endWindowDatetime>beginPreictalDatetime:
                    allWindowsClasses[seizureIndex].append(1) # Preictal
                else:
                    allWindowsClasses[seizureIndex].append(0) # Interictal
    return (allWindowsDataWithoutSph,allWindowsClasses,allWindowsDatetimesWithoutSph)

'''Split the data in train, validation and test using a chronological approach. For example, having
7 seizures and using 40% of the seizures for testing, the method divided the data three times: first
the 1st, 2nd and 3rd for train, 4th for validation and 5th for test, then 1st, 2nd, 3rd and 4th for train,
5th for validation and 6th for test and finally, 1st, 2nd, 3rd, 4th and 5th for train, 6th for validation 
and 7th for test'''
def splitDataTrainValidationTest(allData,trainSeizures,validationSeizures):
    preparedDataset = []
    numberSeizures = len(allData)
    # Get start and end indexes of training seizures
    trainIndexes = [0,trainSeizures]
    # Get last validation seizures' index
    lastValidationSeizureNumber = trainSeizures+validationSeizures
    # Get start and end indexes of validation seizures
    validationIndexes = [trainSeizures,lastValidationSeizureNumber]
    # Get start and end indexes of test seizures
    testIndexes = [lastValidationSeizureNumber,numberSeizures]
    # Get training seizures
    trainData = allData[trainIndexes[0]:trainIndexes[1]]
    # Get validation seizures
    if validationSeizures>0:
        validationData = allData[validationIndexes[0]:validationIndexes[1]]
    # Get test seizures
    testData = allData[testIndexes[0]:testIndexes[1]]
    # Append train, validation and test sets
    if validationSeizures>0:
        preparedDataset.append([trainData,validationData,testData])
    else:
        preparedDataset.append([trainData,testData])
        
    return preparedDataset[0]

'''Join lists of lists'''
def prepareDatasetForModel(dataset,setType):
    if setType == 'train':
        dataset = list(itertools.chain.from_iterable(dataset))
        dataset = np.array(dataset)
    elif setType == 'validation' or setType == 'test':
        newDataset = []
        for subset in dataset:
            subset = np.array(subset)
            newDataset.append(subset)
        if len(newDataset)==1:
            dataset = newDataset[0]
        
    return dataset

'''Convert data from 1D to 2D'''
def reshapeData(data):
    dataShape = list(data.shape)
    numberSamples = dataShape[0]
    numberTemporalSamples = dataShape[1]
    numberChannels = dataShape[2]
    dataShape = [numberSamples,numberTemporalSamples,numberChannels]
    dataShape = tuple(dataShape)
    data = np.reshape(data,dataShape)
    return data

'''Normalize data using max-min normalization'''
def normalizeData(data,trainingData,maxValue=0,minValue=0):
    if trainingData:
        minValue = np.min(data)
        maxValue = np.max(data)
        data = (data - minValue) / (maxValue - minValue)
    else:
        data = (data - minValue) / (maxValue - minValue)
    return (data,maxValue,minValue)

'''Statistical Moments Features'''
def statisticalFeatures(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    skewness = scipy.stats.skew(signal)
    kurtosis = scipy.stats.kurtosis(signal)
    
    return (mean,variance,skewness,kurtosis)

'''Hjorth Parameters: 
    Activity->Variance(Signal)
    Mobility->Sqrt(Variance(First Derivative)/Variance(Signal))
    Complexity->Mobility(First Derivative)/Mobility(Signal)'''
def hjorthParameters(signal):
    firstDerivSignal = np.diff(signal)
    secondDerivSignal = np.diff(signal,2)

    varianceSignal = np.mean(signal ** 2)
    varianceFirstDerivSignal = np.mean(firstDerivSignal ** 2)
    varianceSecondDerivSignal = np.mean(secondDerivSignal ** 2)

    activity = varianceSignal
    mobility = np.sqrt(varianceFirstDerivSignal / varianceSignal)
    complexity = np.sqrt(varianceSecondDerivSignal / varianceFirstDerivSignal) / mobility

    return activity, mobility, complexity

def spectralBandsFeatures(signal,fs=256):
    freqs, psd = periodogram(signal,fs,window='hann',scaling='spectrum')
    deltaBandPowers = psd[((freqs>=0.5) & (freqs<4))]
    deltaBandPower = scipy.integrate.simps(deltaBandPowers)
    
    thetaBandPowers = psd[((freqs>=4) & (freqs<8))]
    thetaBandPower = scipy.integrate.simps(thetaBandPowers)
    
    alphaBandPowers = psd[((freqs>=8) & (freqs<13))]
    alphaBandPower = scipy.integrate.simps(alphaBandPowers)
    
    betaBandPowers = psd[((freqs>=13) & (freqs<30))]
    betaBandPower = scipy.integrate.simps(betaBandPowers)
    
    gammaOneBandPowers = psd[((freqs>=30) & (freqs<=49))]
    gammaOneBandPower = scipy.integrate.simps(gammaOneBandPowers)
    
    gammaTwoBandPowers = psd[((freqs>=51) & (freqs<70))]
    gammaTwoBandPower = scipy.integrate.simps(gammaTwoBandPowers)
    
    gammaThreeBandPowers = psd[((freqs>=70) & (freqs<90))]
    gammaThreeBandPower = scipy.integrate.simps(gammaThreeBandPowers)
    
    totalBandPowers = psd[((freqs>=0.5) & (freqs<=90))]
    totalBandPower = scipy.integrate.simps(totalBandPowers)
    
    relativeDeltaBandPower = deltaBandPower/totalBandPower
    relativeThetaBandPower = thetaBandPower/totalBandPower
    relativeAlphaBandPower = alphaBandPower/totalBandPower
    relativeBetaBandPower = betaBandPower/totalBandPower
    relativeGammaOneBandPower = gammaOneBandPower/totalBandPower
    relativeGammaTwoBandPower = gammaTwoBandPower/totalBandPower
    relativeGammaThreeBandPower = gammaThreeBandPower/totalBandPower
    
    return [deltaBandPower,thetaBandPower,alphaBandPower,betaBandPower,
            gammaOneBandPower,gammaTwoBandPower,gammaThreeBandPower,
            relativeDeltaBandPower,relativeThetaBandPower,relativeAlphaBandPower,
            relativeBetaBandPower,relativeGammaOneBandPower,relativeGammaTwoBandPower,
            relativeGammaThreeBandPower]

def spectralEdgeFrequencyFeatures(signal,fs=256):
    freqs, power = periodogram(signal,fs,window='hann',scaling='spectrum')
    power_cum = scipy.integrate.cumtrapz(power)
    sef50Idx = (np.abs(power_cum - 0.5*scipy.integrate.trapz(power))).argmin() # closest freq holding 50% spectral power
    sef75Idx = (np.abs(power_cum - 0.75*scipy.integrate.trapz(power))).argmin() # closest freq holding 75% spectral power
    sef90Idx = (np.abs(power_cum - 0.9*scipy.integrate.trapz(power))).argmin() # closest freq holding 90% spectral power
    sef50 = freqs[sef50Idx]
    sef75 = freqs[sef75Idx]
    sef90 = freqs[sef90Idx]
    return [sef50,sef75,sef90]
    
def waveletFeatures(signal,motherWavelet):
    coeffs = pywt.wavedec(signal, motherWavelet)
    
    d1Energy = np.sum(np.power(np.abs(coeffs[-1]), 2))
    d2Energy = np.sum(np.power(np.abs(coeffs[-2]), 2))
    d3Energy = np.sum(np.power(np.abs(coeffs[-3]), 2))
    d4Energy = np.sum(np.power(np.abs(coeffs[-4]), 2))
    d5Energy = np.sum(np.power(np.abs(coeffs[-5]), 2))
    d6Energy = np.sum(np.power(np.abs(coeffs[-6]), 2))
    d7Energy = np.sum(np.power(np.abs(coeffs[-7]), 2))
    d8Energy = np.sum(np.power(np.abs(coeffs[-8]), 2))
    totalEnergy = d1Energy+d2Energy+d3Energy+d4Energy+d5Energy+d6Energy+d7Energy+d8Energy
    d1Energy/=totalEnergy
    d2Energy/=totalEnergy
    d3Energy/=totalEnergy
    d4Energy/=totalEnergy
    d5Energy/=totalEnergy
    d6Energy/=totalEnergy
    d7Energy/=totalEnergy
    d8Energy/=totalEnergy
    
    return [d1Energy,d2Energy,d3Energy,d4Energy,d5Energy,
            d6Energy,d7Energy,d8Energy]

def calculateWindowFeatures(signalWindows):
    numberChannels = signalWindows.shape[2]
    allSignalWindowsFeatures = []
    for index,signalWindow in  enumerate(signalWindows):
        signalWindowFeatures = []
        signalWindowFeaturesChannel = []
        for channelIndex in range(numberChannels):
            signalChannel = signalWindow[:,channelIndex]
            channelMean,channelVariance,channelSkewness,channelKurtosis = statisticalFeatures(signalChannel)
            _,channelHjorthMobility,channelHjorthComplexity = hjorthParameters(signalChannel)
            
            channelDeltaPow,channelThetaPow,channelAlphaPow,\
                channelBetaPow,channelGammaOnePow,channelGammaTwoPow,\
                    channelGammaThreePow,channelRelDeltaPow,channelRelThetaPow,\
                        channelRelAlphaPow,channelRelBetaPow,channelRelGammaOnePow,\
                            channelRelGammaTwoPow,channelRelGammaThreePow = spectralBandsFeatures(signalChannel)
            waveletCoefsEnergy = waveletFeatures(signalChannel, 'db4')
            channelSef50,channelSef75,channelSef90 = spectralEdgeFrequencyFeatures(signalChannel)
            signalWindowFeaturesChannel = [channelMean,channelVariance,channelSkewness,channelKurtosis,
                                            channelHjorthMobility,channelHjorthComplexity,
                                            channelRelDeltaPow,channelRelThetaPow,
                                            channelRelAlphaPow,channelRelBetaPow,channelRelGammaOnePow,
                                            channelRelGammaTwoPow,channelRelGammaThreePow,
                                            channelDeltaPow/channelThetaPow,channelDeltaPow/channelAlphaPow,
                                            channelDeltaPow/channelBetaPow,channelDeltaPow/channelGammaOnePow,
                                            channelDeltaPow/channelGammaTwoPow,channelDeltaPow/channelGammaThreePow,
                                            channelThetaPow/channelAlphaPow,channelThetaPow/channelBetaPow,
                                            channelThetaPow/channelGammaOnePow,channelThetaPow/channelGammaTwoPow,
                                            channelThetaPow/channelGammaThreePow,channelAlphaPow/channelBetaPow,
                                            channelAlphaPow/channelGammaOnePow,channelAlphaPow/channelGammaTwoPow,
                                            channelAlphaPow/channelGammaThreePow,channelBetaPow/channelGammaOnePow,
                                            channelBetaPow/channelGammaTwoPow,channelBetaPow/channelGammaThreePow,
                                            channelGammaOnePow/channelGammaTwoPow,channelGammaOnePow/channelGammaThreePow,
                                            channelGammaTwoPow/channelGammaThreePow]
            signalWindowFeaturesChannel.extend(waveletCoefsEnergy)
            signalWindowFeatures.append(signalWindowFeaturesChannel)
        signalWindowFeatures = np.array(signalWindowFeatures)
        allSignalWindowsFeatures.append(signalWindowFeatures)
    
    return np.array(allSignalWindowsFeatures)
                    

def from2Dto3DEEG(data):
    nrSamples = len(data)
    newData = []
    
    for sample in data:
        newSample = from2Dto3DSample(sample)
        newData.append(newSample)
    
    return np.array(newData)
        
def from2Dto3DSample(sample):
    
    newSample = []
    
    for timestep in sample:
        newTimestep = np.zeros((5,5,1))
        newTimestep[1,0,0] = timestep[10,0]
        newTimestep[2,0,0] = timestep[12,0]
        newTimestep[3,0,0] = timestep[14,0]
        newTimestep[0,1,0] = timestep[0,0]
        newTimestep[1,1,0] = timestep[2,0]
        newTimestep[2,1,0] = timestep[4,0]
        newTimestep[3,1,0] = timestep[6,0]
        newTimestep[4,1,0] = timestep[8,0]
        newTimestep[1,2,0] = timestep[16,0]
        newTimestep[2,2,0] = timestep[17,0]
        newTimestep[3,2,0] = timestep[18,0]
        newTimestep[0,3,0] = timestep[1,0]
        newTimestep[1,3,0] = timestep[3,0]
        newTimestep[2,3,0] = timestep[5,0]
        newTimestep[3,3,0] = timestep[7,0]
        newTimestep[4,3,0] = timestep[9,0]
        newTimestep[1,4,0] = timestep[11,0]
        newTimestep[2,4,0] = timestep[13,0]
        newTimestep[3,4,0] = timestep[15,0]
        newSample.append(newTimestep)
    
    return np.array(newSample)

