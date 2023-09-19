"""
Created on Wed Nov  9 13:52:03 2022

@author: juliana
"""

# %% Imports

import os
import glob
from utils import *
import pandas as pd
from dateutil import parser
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# %% Import data - EPILEPSIAE

folder_name = 'Clusters'

path_folders = os.path.join(os.getcwd(), folder_name,'pat*')

folders_list = glob.glob(path_folders)

gridsearch_csv = pd.read_csv('gridsearch.csv', ';', header=None)

new_folder_name = 'Graphs'

create_new_folder(new_folder_name)

new_path = os.path.join(os.getcwd(),new_folder_name)


for path in folders_list:
    
    patient_seizure = path.split('/')[-1]
    
    patient_id = patient_seizure.split('_')[1]
    seizure_id = patient_seizure.split('_')[3].split('.')[0]
    
    
    print('\n Patient :', patient_seizure, '\n')
    
    seizure = pd.read_csv(path).drop(columns=['Unnamed: 0'])
    datetime =  seizure[['Datetime']].values
    
    array_size = seizure.shape[0]
    
    x = [i[0] for i in seizure[['Axis x']].values]
    y = [i[0] for i in seizure[['Axis y']].values]
    z = [i[0] for i in seizure[['Axis z']].values]

    time = [parser.parse(i[0]).strftime("%Hh%M") for i in datetime] 
    
    c = np.arange(array_size)
    
    time_ticks = [time[0], time[int(array_size/4)], time[int(array_size/2)], time[int(3*array_size/4)],time[-1]]
    c_ticks = [c[0], c[int(array_size/4)], c[int(array_size/2)], c[int(3*array_size/4)], c[-1]]
    
    label_kmeans = [i[0] for i in seizure[['KMeans']].values]
    label_ah = [i[0] for i in seizure[['AH']].values]
    label_hdbscan = [i[0] for i in seizure[['HDBSCAN']].values]
    
    label = (seizure[['AE - AH']].values)

    label_filtered = pd.DataFrame(label).rolling(9, min_periods=1, center=True).mean().round(0).to_numpy()
    label_filtered = [int(i[0]) for i in label_filtered]

    label = [i[0] for i in label]
    
    fig = make_subplots(rows=3, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}],[{'type': 'scene'}, {'type': 'scene'}], [{'type': 'xy', 'colspan': 2}, None]],
                        subplot_titles=( 'Dimensionality Reduction - AE', 'Kmeans', 'Agglomerative', 'HDBSCAN', 'Classes over time'),
                        shared_yaxes=False, shared_xaxes=True, horizontal_spacing=0.05, vertical_spacing=0.1)

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, 
                               marker=dict(size=3, color=c, colorscale='Viridis', 
                               showscale=True, colorbar=dict(len=0.48,  x=1, y=0.85,
                                                             tickvals=c_ticks, ticktext=time_ticks, 
                                                             ticks='outside', tickmode='array', title="Seizure onset")), 
                               mode='markers'),row=1, col=1)
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, 
                               marker=dict(size=3, color=label_kmeans, colorscale='Viridis'), 
                               mode='markers'),row=1, col=2)
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, 
                               marker=dict(size=3, color=label_ah, colorscale='Viridis'), 
                               mode='markers'),row=2, col=1)
    
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, 
                               marker=dict(size=3, color=label_hdbscan, colorscale='Viridis'), 
                               mode='markers'),row=2, col=2)
    
    
    fig.add_trace(go.Scatter(x=time, y=label_filtered, mode='lines+markers', name='clustering solution',
                              line=dict(dash='dash', color='rgb(189,189,189)')), row=3, col=1)

    fig.add_trace(go.Scatter(x=time, y=label_filtered, mode='markers',
                              marker=dict(color=label, colorscale='Viridis')), row=3, col=1)
    
    fig.update_xaxes(tickformat = '%Hh%M', row=3, col=1)
    

    fig.update_layout(showlegend=False, title_text='Patient: ' + str(patient_id) +
                      ' Seizure: '+str(seizure_id), margin=dict(l=80, r=10, t=70, b=20))

    fig.show()

    fig.write_html(os.path.join(new_path, 'pat_' + str(patient_id) +'_seizure_' + str(seizure_id) + '.html'))
    

    

