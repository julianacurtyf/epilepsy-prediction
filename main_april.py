"""
Created on Wed Nov  9 13:52:03 2022

@author: juliana
"""

# %% Imports

from sklearn.cluster import AgglomerativeClustering
import os
import numpy as np
from keras.activations import *
from keras.optimizers import Adam
import pandas as pd
from utils import get_sorted_folders, get_files, get_patient_and_seizure_id, preprocessing, create_new_folder, create_file, get_cluster_interval, dunn_fast, target_distribution, get_time_from_idx
from DeepClusteringLayer import ClusteringLayer
from tensorflow.keras.models import Model
from sklearn.model_selection import ParameterGrid
import gc
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Flatten, Dropout, UpSampling1D, Reshape

os.environ["CUDA_VISIBLE_DEVICES"] = "9"


# %% Import data - EPILEPSIAE

path_to_raw_data = '/path_to_data/Datasets/'

folders_list = get_sorted_folders(path_to_raw_data)

folder_name = 'AE models'
    
create_new_folder(folder_name)

gridsearch_csv = pd.read_csv('gridsearch.csv', ';', header=None)

filename = 'metrics_trained_model.csv'
header = ['idx','parameters', 'dunn_idx', 'density', 'begin', 'end']

writer, file = create_file(os.path.join(os.getcwd(),folder_name), filename, header, file_type='csv')


for j in range(len(folders_list)):
    
    seizure_information, data, datetimes = get_files(folders_list[j], 'raw')
    
    patient_id = get_patient_and_seizure_id(folders_list[j], 'raw')
    
    data, datetime = preprocessing(data, datetimes)
    
    numberSeizures = len(data)

    assert(patient_id == eval(gridsearch_csv[0][j]))

    parameters = eval(gridsearch_csv[2][j])
    idx = eval(gridsearch_csv[1][j])
    
    train_set = np.concatenate((data[idx[0]],data[idx[1]]))
    validation_set = data[idx[2]]
    target_train = np.concatenate((datetime[idx[0]],datetime[idx[1]]))
    target_val = datetime[idx[2]]
    
    mean_train_set = np.mean(train_set)
    std_train_set = np.std(train_set)
    train_set = (train_set - mean_train_set)/std_train_set
    validation_set = (validation_set - mean_train_set)/std_train_set
    
    # %% AE
    
    input_dim = (train_set.shape[1], train_set.shape[2])
    
    n_clusters = 2
    dist_metric = 'eucl'
    dimensions = 2
    

    loss = 'mse'
    metrics = ['mse']
    optimizer = 'adam'
    epochs = 10
    batch_size = 32
    layer = 'decoded'
    function = 'swish'

    print(parameters)

    n_filters = parameters['n_filters']
    kernel_size = parameters['kernel_size']
    pool_size = parameters['pool_size']
    function = parameters['activation_function']

    x = Input(shape=input_dim, name='input')
    
    encoded = Conv1D(n_filters, kernel_size, strides=pool_size, padding='same', activation=function)(x)
    encoded = BatchNormalization()(encoded)
    
    encoded = Conv1D(n_filters, kernel_size, strides=pool_size, padding='same', activation=function)(encoded)
    encoded = BatchNormalization()(encoded)
    
    encoded = Flatten(name='flatten')(encoded)
    encoded = Dropout(0.5)(encoded)
    
    latent = Dense(3, activation=function, name='latent')(encoded)
    
    decoded = Dense(encoded.shape[1], activation=function)(latent)
    decoded = Reshape((-1, n_filters), name='reshape')(decoded) # dense -> muda de 1 pra n_filters
    
    decoded = UpSampling1D(pool_size)(decoded)
    decoded = Conv1D(n_filters, kernel_size, strides=1, padding='same', activation=function)(decoded)
    
    decoded = UpSampling1D(pool_size)(decoded)
    decoded = Conv1D(train_set.shape[2], kernel_size, strides=1, padding='same', activation='linear')(decoded)
    
    
    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')
    
    # Encoder model
    encoder = Model(inputs=x, outputs=latent, name='encoder')
    
    autoencoder.summary()
    
    autoencoder.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    
    history = autoencoder.fit(train_set, train_set, epochs=epochs, batch_size=batch_size, verbose=1)
    
    features = encoder.predict(train_set)
    
    print('Initializing Clustering...')
    
    hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(features)
    
    cluster_centers = np.array( [features[hc.labels_ == c].mean(axis=0) for c in range(n_clusters)])
    
    clusteringLayer = ClusteringLayer(n_clusters, dist_metric=dist_metric, dimensions=2, name='clustering')(encoder.output)
    
    deepClusteringAutoencoder = Model(
        autoencoder.input, [autoencoder.output, clusteringLayer])
    
    deepClusteringAutoencoder.get_layer(
        name='clustering').set_weights([cluster_centers])
    
    deepClusteringAutoencoder.compile(optimizer=Adam(
        learning_rate=1e-3), loss=['mse', 'bce'])
    
    print('Training Deep Clustering Network...')
    
    batch_size = 64
    epochs = 2000  
    evaluate_interval = 10
    clean_interval = 5
    train_samples = len(train_set)
    last_train_pred = None
    stop_training = False
    training_tolerance = 0.0001
    assignment_changes = 0
    patience = 15
    patience_count = 0
    es = EarlyStopping(mode='min', verbose=1, patience=int(epochs*0.1),restore_best_weights=True) 
    
    for epoch in range(epochs):
    
        print('Epoch %d/%d' % (epoch + 1, epochs))
        _, q = deepClusteringAutoencoder.predict(train_set, verbose=0)
        p = target_distribution(q)
        
        _, q_val = deepClusteringAutoencoder.predict(validation_set, verbose=0)
        p_val = target_distribution(q_val)
    
        if epoch % evaluate_interval == 0:
    
            train_pred = q.argmax(axis=1)
    
            if last_train_pred is not None:
                assignment_changes = np.sum(last_train_pred != train_pred).astype(
                    np.float32) / train_pred.shape[0]
    
            last_train_pred = train_pred
    
            if epoch > 0 and assignment_changes < training_tolerance:
                patience_count += 1
    
                if patience_count >= patience:
                    stop_training = True
            else:
                patience_count = 0
    
            print(f'Assignment Changes: {assignment_changes}')
    
        deepClusteringAutoencoder.fit(
            train_set, [train_set, p], batch_size=batch_size, epochs=1, verbose=1,callbacks=[es], validation_data=(validation_set, [validation_set,p_val]))
    
        if epoch % clean_interval == 0:
            del q, p
            gc.collect()
    
        if stop_training:
            break
    
    
    # %% Validation 
    
    seizure_number = idx[2]
    
    val_reconstruction, val_pred = deepClusteringAutoencoder.predict(validation_set)
    val_pred = np.argmax(val_pred, axis=1)
    
    latent = deepClusteringAutoencoder.get_layer('latent').output
    
    latentModel = Model(deepClusteringAutoencoder.input, latent)
    
    reduced_df = encoder.predict(validation_set)
    
    preictal_begin_idx, preictal_end_idx, preictal_density = get_cluster_interval(train_pred, 6)
    
    preictal_begin, preictal_end = get_time_from_idx(datetime[seizure_number], preictal_begin_idx, preictal_end_idx)
    
    dunn_index = dunn_fast(reduced_df, val_pred)
    
    writer.writerow([idx, parameters, dunn_index, preictal_density, preictal_begin_idx, preictal_end_idx])

    # %% Saving 

    folder_name = 'AE models'

    create_new_folder(folder_name)

    deepClusteringAutoencoder.save(os.path.join(os.getcwd(), folder_name, 'pat_' + str(patient_id) +'_seizure_' + str(seizure_number)))

file.close()