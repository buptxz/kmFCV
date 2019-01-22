#!/usr/bin/env python
# coding: utf-8

# # Evaluation of extrapolation performances of materials properties prediction
import argparse
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    
import math
import pandas as pd
import numpy as np
import os
import sys
import csv
import subprocess

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, ShuffleSplit, KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, clone
from sklearn.neighbors import KNeighborsRegressor

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from ase.io import read

parser = argparse.ArgumentParser(description='kmFCV')
parser.add_argument('--data-path', default='data',
                    help='dataset options, started with the path to root dir, '
                    'then other options')
parser.add_argument('--demo', action='store_true', 
                    help='Quick demo mode, 1000 samples')
parser.add_argument('--dataset', choices=['mp', 'supercon'],
                    default='mp', help='dataset name (default: mp)')
parser.add_argument('--property', choices=['formation_energy', 'band_gap', 'Tc'],
                    default='formation_energy', help='property name (default: formation_energy)')
parser.add_argument('--feature', choices=['magpie', 'composition', 'ptr'],
                    default='magpie', help='feature name (default: magpie)')
parser.add_argument('--model', choices=['1nn', 'rf', 'mlp', 'cnn', 'cgcnn', 'elemnet', 'svr'],
                    default='rf', help='model name (default: rf)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='epochs to train (default: 30)')
parser.add_argument('--validation', choices=['cv', 'fcv', 'holdout', 'iecv'],
                    default='cv', help='validation method (default: cv)')
parser.add_argument('--split', choices=['random', 'sorted'],
                    default='random', help='split (default: random)')
parser.add_argument('--hybrid', action='store_true', 
                    help='hybrid traning mode')
parser.add_argument('-k', default=5, type=int, metavar='N',
                    help='k value (default: 5)')
parser.add_argument('-m', default=1, type=int, metavar='N',
                    help='m value (default: 1)')

args = parser.parse_args(sys.argv[1:])
data_folder = args.data_path
dataset = args.dataset
pred_property = args.property
feature = args.feature
ml_method = args.model
epochs = args.epochs
quick_demo = args.demo
hybrid = args.hybrid
validation_type = args.validation
split = args.split
k = args.k
m = args.m
if validation_type == 'cv':
    validation_method = '{}_fold_{}'.format(k, validation_type)
else:
    validation_method = '{}_fold_{}_step_{}'.format(k, m, validation_type)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, CuDNNLSTM, Input, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from keras import backend as K
from keras import regularizers

def main():
    validation_types = {
        'cv': cv,
        'fcv': fcv,
        'holdout': holdout,
        'iecv': iecv
    }

    print('-----------------------------------------------------------------------------------------------')
    print('{} dataset, {} property, {} feature, {} method, {} validation'.format(dataset, pred_property, feature, ml_method, validation_method))
    if quick_demo:
        print('Demo mode.')
    # print('k = {}'.format(k))
    # if args.validation == 'fcv':
    #     print('m = {}'.format(m))
    print('-----------------------------------------------------------------------------------------------')
    
    if ml_method == 'cgcnn':
        subprocess.run(['python', 'cgcnn_main.py', '--demo', str(1) if quick_demo else str(0), '--hybrid', str(1) if hybrid else str(0), 
            '{}/cgcnn_{}'.format(data_folder, pred_property), '--validation', validation_type, '-k', str(k), '--epochs', str(epochs)])
        result = pd.read_csv('cgcnn_result.csv', names=['prediction', 'target'])
        evaluation_plot(result.target, result.prediction)
    else:
        y = None
        try:
            y = np.load('{}/features/{}_{}_y.npy'.format(data_folder, dataset, pred_property))
        except:
            pass
        
        if y is not None:
            try:
                X = pd.read_csv('{}/features/{}_{}_{}_X.csv'.format(data_folder, dataset, pred_property, feature))
            except:
                X = np.load('{}/features/{}_{}_{}_X.npy'.format(data_folder, dataset, pred_property, feature))
            
            if quick_demo:
                if isinstance(X, pd.DataFrame):
                    X = X.head(1000)
                else:
                    X = X[:1000]
                y = y[:1000]
        
            # set the memory usage
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
            # import logging
            # logging.getLogger("tensorflow").setLevel(logging.WARNING)
            from keras.backend.tensorflow_backend import set_session
            import tensorflow as tf
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            set_session(tf.Session(config=tf_config))
            tf.logging.set_verbosity(tf.logging.ERROR)

            if ml_method == '1nn':
                validation_types[validation_type](KNeighborsRegressor(1, n_jobs=-1), X, y, k=k)

            if ml_method == 'rf':
                validation_types[validation_type](RandomForestRegressor(100, max_features=10, n_jobs=-1), X, y)

            if ml_method == 'svr':
                validation_types[validation_type](GridSearchCV(SVR(gamma='scale'),
                                                                param_grid=dict(C=[0.1], epsilon =[0.01, 0.1]),
                                                                scoring='neg_mean_absolute_error', verbose=2, cv=ShuffleSplit(5), n_jobs=-1), 
                                                                X, y)
            
            if ml_method == 'mlp':
                validation_types[validation_type](mlp, X, y, shape=X.shape[1])

            if ml_method == 'elemnet':
                validation_types[validation_type](elemnet, X, y, shape=X.shape[1])

            if ml_method == 'cnn':
                validation_types[validation_type](cnn, X, y, shape=(X.shape[1], X.shape[2], X.shape[3]))

            if ml_method == 'lstm':
                validation_types[validation_type](lstm, X, y, shape=(X.shape[0], X.shape[1]))


# ## 1. Get data (Material Project, icsd subset of oqmd, Superconductivity)

# ### Material Project

# In[4]:


def get_mp(properties=[]):   
    mp_df = pd.read_csv("{}/mp_raw.csv".format(data_folder), nrows=1000 if quick_demo else None)
    
#     # Read formula string as a dictionary
#     mp_df['formula_dict'] = mp_df['formula'].transform(lambda x: Composition(x).as_dict())
    
    formula_dict_list = []
    for i in range(mp_df['formula'].count()):
        try:
            formula_dict_list.append(Composition(mp_df['formula'][i]).as_dict())
        except:
            formula_dict_list.append(np.nan)
    mp_df['formula_dict'] = pd.Series(formula_dict_list)
    
    mp_df = mp_df.dropna(subset=['formula_dict'])
    
    existing_properties = mp_df.columns.values.tolist()
    new_properties = list(set(properties) - set(existing_properties))
        
    if len(new_properties) > 0:
        mp_new_properties = download_mp(new_properties)
        if len(mp_df) == len(mp_new_properties):
            mp_df = pd.concat([mp_df, mp_new_properties], axis=1, join='outer')
        else:
            raise Exception('Mismatch row number')
        mp_df.to_csv("mp_raw.csv", index=False)
 
    if len(properties) > 0:
        return mp_df[properties]
    else:
        return mp_df

def download_mp(properties):
    # Material Project key
    mpdr = MPDataRetrieval(api_key='8TmS5ERNxWtrzo2K')

    energy_min = -5
    energy_max = 5
    energy_step = 0.02

    mp_df = pd.DataFrame()
    energy = energy_min
    while energy <= energy_max:
        mp_df_subquery = mpdr.get_dataframe(
            criteria={"formation_energy_per_atom": {"$gt": energy, "$lte": energy + energy_step}}, 
            properties=list(set(properties) + set(['formation_energy_per_atom']))
        )
        mp_df = pd.concat([mp_df, mp_df_subquery])
        print("There are {} entries for range {} {}, total {}".format(mp_df_subquery['formation_energy_per_atom'].count(), 
                                                                      energy, 
                                                                      energy+0.1, 
                                                                      mp_df['formation_energy_per_atom'].count()))
        energy = energy + energy_step

    mp_df.reset_index(drop=True, inplace=True)
    return mp_df[properties]


# ### ICSD/OQMD

# In[5]:


def get_icsd(path = '{}/voro-ml-si/datasets/icsd-all'.format(data_folder)):
    oqmd_df = pd.read_csv('{}/properties.txt'.format(path), sep=' ', nrows=1000 if quick_demo else None)
    oqmd_df['formula'] = oqmd_df['filename'].transform(lambda x: x.split('-')[1])
    oqmd_df['structure'] = oqmd_df['filename'].transform(
        lambda x: AseAtomsAdaptor().get_structure(read(os.path.join(path, x), format='vasp')))     
    oqmd_df['formula_dict'] = oqmd_df['formula'].transform(lambda x: Composition(x).as_dict())
    oqmd_df['nelements'] = oqmd_df['formula'].transform(lambda x: len(Composition(x).as_dict().keys()))
    oqmd_df=oqmd_df.rename(columns = {'delta_e':'formation_energy'})
    oqmd_df=oqmd_df.rename(columns = {'bandgap':'band_gap'})
    return oqmd_df


# ### Superconductivity

# In[6]:


def get_supercon():
    supercon_df = pd.read_csv('{}/Supercon_data.csv'.format(data_folder), nrows=1000 if quick_demo else None)
    supercon_df.columns = ['formula', 'Tc']
    supercon_df['formula_dict'] = supercon_df['formula'].transform(lambda x: Composition(x).as_dict())
    supercon_df['nelements'] = supercon_df['formula'].transform(lambda x: len(Composition(x).as_dict().keys()))
    return supercon_df


# ### AFLOW

# In[7]:


def get_aflow(path = '{}/aflow'.format(data_folder)):
    pass
    
def download_aflow():
    from aflow import search
    import urllib.request
    
    aflow_data = search(batch_size=100).select(K.Egap).filter(K.Egap > 0).orderby(K.Egap, reverse=True)
    count = 0
    aflow_band_gap = np.zeros(len(result))
    aflow_compound = []
    
    for entry in aflow_data:
        sys.stdout.write("Downloading: %d out of %d  \r" % (count, len(aflow_data)))
        sys.stdout.flush()
        try:
            url = str(entry) + '/' + str(entry).split("/")[-1] + '.cif'
            file_path = '{}/{}_{}_{}.cif'.format(path, count, str(entry).split("/")[-2], str(entry).split("/")[-1])
            urllib.request.urlretrieve(url, file_path)
        except:
            pass

        aflow_band_gap[count] = entry.Egap
        aflow_compound.append(entry.compound)
        count += 1
    
    aflow_df = pd.DataFrame({'full_formula':aflow_compound[0:33762],'band_gap':aflow_band_gap[0:33762]})
    aflow_df.head()
    aflow_df.to_csv('aflow/aflow.csv')


# ## 2. Data preprocessing

# Selected elements using perioic table representation

# In[8]:


ptr_dict = {'Li': (0,0), 'Be': (0,1), 'B': (0,12), 'C': (0,13), 'N': (0,14), 'O': (0,15), 'Na': (1,0),
           'Mg': (1,1), 'Al': (1,12), 'Si': (1,13), 'P': (1,14), 'S':(1,15), 'K':(2,0), 'Ca':(2,1), 'Sc':(2,2), 
            'Ti':(2,3), 'V':(2,4), 'Cr':(2,5), 'Mn':(2,6), 'Fe':(2,7), 'Co':(2,8), 'Ni':(2,9), 'Cu':(2,10), 'Zn':(2,11), 
            'Ga':(2,12), 'Ge':(2,13), 'As':(2,14), 'Se':(2,15), 'Rb':(3,0), 'Sr':(3,1), 'Y':(3,2), 'Zr':(3,3), 
            'Nb':(3,4), 'Mo':(3,5), 'Tc':(3,6), 'Ru':(3,7), 'Rh':(3,8), 'Pd':(3,9), 'Ag':(3,10), 'Cd':(3,11), 
            'In':(3,12), 'Sn':(3,13), 'Sb':(3,14), 'Te':(3,15), 'Cs':(4,0), 'Ba':(4,1), 'Hf':(4,3), 'Ta':(4,4), 
            'W':(4,5), 'Re':(4,6), 'Os':(4,7), 'Ir':(4,8), 'Pt':(4,9), 'Au':(4,10), 'Hg':(4,11), 'Ti':(4,12), 
            'Pb':(4,13), 'Bi':(4,14), 'Po':(4,15)}


# In[9]:


ptr_atom_type = ptr_dict.keys()
ptr_atom_type


# Preprocessing

# In[10]:


def data_preprocessing(input_data, 
                       properties_to_predict,
                       valid_range=None,
                       allowed_element=ptr_atom_type,
                       formula_column='formula',
                       remove_duplicate=True, remove_outliers=True, 
                       remove_ill_converged=True, remove_one_element_compound=True,
                       inplace=False):
    if not inplace:
        data = input_data.copy()
    else:
        data = input_data
    
    # Get only the groundstate and each composition
    if remove_duplicate:
        original_count = len(data)
        data.sort_values('formation_energy', ascending=True, inplace=True)
        data.drop_duplicates(formula_column, keep='first', inplace=True)
        print('Remove duplicate composition: removed %d/%d entries'%(original_count - len(data), original_count))
    
    # Sort by predicted property
    data.sort_values(properties_to_predict, ascending=True, inplace=True)

    # Filter
    if properties_to_predict == 'band_gap':
        original_count = len(data)
        data = data[data[properties_to_predict] > 0]
        print('Remove 0 band gap samples: removed %d/%d entries'%(original_count - len(data), original_count))
    if properties_to_predict == 'Tc':
        original_count = len(data)
        data = data[data[properties_to_predict] >= 10]
        print('Remove < 10 Tc samples: removed %d/%d entries'%(original_count - len(data), original_count))
    
    # Remove outliers
    if remove_outliers:
        original_count = len(data)
        data = data[np.abs(data[properties_to_predict]-data[properties_to_predict].mean()) <= (5*data[properties_to_predict].std())]
        # data = data[np.logical_and(data[properties_to_predict] >= -20, data['formation_energy_per_atom'] <= 5)]
        print('Remove outliers: removed %d/%d entries'%(original_count - len(data), original_count))
    
    # Remove ill converged samples
    if remove_ill_converged:
        try:
            original_count = len(data)
            data = data[data.has_bandstructure == True]
            data = data[data['elasticity.warnings'].isnull()]
            print('Remove ill converged samples: removed %d/%d entries'%(original_count - len(data), original_count))
        except:
            pass
    
    # Remove one element compound
    if remove_one_element_compound:
        if 'nelements' not in data.columns:
            data['nelements'] = data[formula_column].transform(lambda x: len(Composition(x).as_dict()))
        original_count = len(data)
        data = data[data.nelements > 1]
        print('Remove one element sample: removed %d/%d entries'%(original_count - len(data), original_count))
    
    data.reset_index(drop=True, inplace=True)
    
    if allowed_element is not None:
        if 'formula_dict' not in data.columns:
            data['formula_dict'] = data[formula_column].transform(lambda x: Composition(x).as_dict())
        original_count = len(data)
        valid_compound = []
        for i in range(data['formula_dict'].count()):
            if_valid = True
            for element in data['formula_dict'][i].keys():
                if element not in allowed_element:
                    if_valid = False
                    break
            valid_compound.append(if_valid)
        data['if_valid'] = pd.Series(valid_compound)
        data = data[data.if_valid == True]
        data = data.drop(columns=['if_valid'])
        print('Remove by element filter: removed %d/%d entries'%(original_count - len(data), original_count))
    
    data.reset_index(drop=True, inplace=True)
    print('Final sample count: {}'.format(data.shape[0]))
    
    if inplace:
        return None
    else:
        return data


# Prepare data for CGCNN. Save the structure as cif file, save the properties as a csv file

# In[11]:


def mp_to_cgcnn(data, property_name, cgcnn_location, folder_name):
    if not os.path.exists('{}/cgcnn/data/{}'.format(cgcnn_location, folder_name)):
        os.mkdir('{}/cgcnn/data/{}'.format(cgcnn_location, folder_name))

    # Generate cif for cgcnn
    for i in range(len(data)):
        Structure.to(Structure.from_str(data['cif'][i], 'cif'), fmt='cif', 
                     filename='{}/cgcnn/data/{}/{}.cif'.format(cgcnn_location, folder_name, i))

    # Generate csv for cgcnn
    data[property_name].to_csv('{}/cgcnn/data/{}/id_prop.csv'.format(cgcnn_location, folder_name), header=False)
    
    # Copy atom properties
    os.system('cp {}/cgcnn/data/sample-regression/atom_init.json ./cgcnn/data/{}/atom_init.json'.format(cgcnn_location, folder_name))


# In[ ]:


def oqmd_to_cgcnn():
    pass


# ## 3. Feature calculation mothod

# ### Magpie feature

# we use the "general-purpose" attributes of [Ward et al 2016](https://www.nature.com/articles/npjcompumats201628).

# In[ ]:


def get_magpie_feature(input_data, formula_column, inplace=False):
    if inplace:
        data = input_data
    else:
        data = input_data[[formula_column]].copy()
    
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                          cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    # Get the feature names
    feature_labels = feature_calculators.feature_labels()
    
    # Compute the features
    data = StrToComposition(target_col_id='composition').featurize_dataframe(data, formula_column, 
                                                                             ignore_errors=True)
    data = feature_calculators.featurize_dataframe(data, col_id='composition')
    print('Generated %d features'%len(feature_labels))
    print('Dataset size:', 'x'.join([str(x) for x in data[feature_labels].shape]))
    
    # Remove entries with NaN or infinite features
    original_count = len(data)
    data = data[~ data[feature_labels].isnull().any(axis=1)]
    print('Removed %d/%d entries'%(original_count - len(data), original_count))
    
    if inplace:
        return None
    else:
        return data[feature_labels]


# ### Composition feature

# In[ ]:


def get_composition_feature(input_data, formula_column, inplace=False):
    if inplace:
        data = input_data
    else:
        data = input_data[[formula_column]].copy()
    
    # Get all atom types
    atom_types = set()
    for formula in data[formula_column].values:
        for atom in Composition(formula).as_dict().keys():
            atom_types.add(atom)
    atom_types = list(atom_types)
    print("{} atom types: {}".format(len(atom_types), atom_types))
    
    # Encode by the fraction of elements
    for atom in atom_types:
        progress = atom_types.index(atom) / len(atom_types) * 100
        sys.stdout.write("Processing progress: %f%% %s  \r" % (progress, atom))
        sys.stdout.flush()
        data[atom] = data[formula_column].transform(lambda x: Composition(x).get_atomic_fraction(atom))
    
    if inplace:
        return data
    else:
        return data[list(atom_types)]


# ### PTR feature

# In[ ]:


def get_ptr_feature(data, formula_column):
    # Raw periodic table matrix
    ptr_matrix_raw = np.full((5, 16), -1.0)
    ptr_matrix_raw[0:2, 2:12] = 0
    ptr_matrix_raw[4,2] = 0
    ptr_matrix_raw
    
    ptr_matrix_list = []
    
    for formula in data[formula_column].values:
        ptr_matrix = np.copy(ptr_matrix_raw)

        for atom in Composition(formula).as_dict().keys():
            ptr_matrix[ptr_dict[atom][0]][ptr_dict[atom][1]] = Composition(formula).as_dict()[atom]

        ptr_matrix_list.append(ptr_matrix)
    
    ptr_matrix_list = np.array(ptr_matrix_list)
    ptr_matrix_list = ptr_matrix_list.reshape(ptr_matrix_list.shape[0], 
                                              1, 
                                              ptr_matrix_list.shape[1], 
                                              ptr_matrix_list.shape[2])
    
    return ptr_matrix_list
    


# ## 4. Model

# ### MLP

# In[ ]:


def mlp(input_dim):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    Dropout(0.5, noise_shape=None, seed=None)
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    Dropout(0.5, noise_shape=None, seed=None)
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    Dropout(0.5, noise_shape=None, seed=None)
    model.add(Dense(1, kernel_initializer='normal'))
    # model.add(Dense(1, kernel_initializer='normal', activity_regularizer=regularizers.l1(0.0005)))

    model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='Adam')
    
    return model

def elemnet(input_dim, l1_reg=True):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1024, kernel_initializer='normal', activation='relu'))
    Dropout(0.8, noise_shape=None, seed=None)
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    Dropout(0.9, noise_shape=None, seed=None)
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    Dropout(0.7, noise_shape=None, seed=None)
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    Dropout(0.8, noise_shape=None, seed=None)
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(32, kernel_initializer='normal', use_bias=False))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    if not l1_reg:
        model.add(Dense(1, kernel_initializer='normal'))
    else:
        model.add(Dense(1, kernel_initializer='normal', activity_regularizer=regularizers.l1(0.001)))

    model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer='Adam')
    
    return model

def lstm(input_shape):
    rnn_unit_size = 256
    dense_unit_size_1 = 128
    dense_unit_size_2 = 90

    model = Sequential()
    input_sequences = Input(shape=input_shape)
    model.add(CuDNNLSTM(rnn_unit_size, return_sequences=False)(input_sequences)) 
    # x = GlobalAveragePooling1D()(processed_sequences)
    model.add(Dense(dense_unit_size_1, activation='relu'))
    model.add(Dense(dense_unit_size_2, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(optimizer="adam", loss='mean_absolute_error', metrics=['mean_absolute_error'])

# ### CNN

# In[ ]:



def cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same',
                input_shape=input_shape, data_format='channels_first'))
    model.add(Conv2D(96, (5, 5), activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(192, activation='relu'))
    model.add(Dense(192, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(loss="mean_absolute_error", optimizer=sgd)
    return model


# ### Random Forest

# In[ ]:


def rf():
    print("Grid search...")
    model = GridSearchCV(RandomForestRegressor(n_estimators=20 if quick_demo else 150, n_jobs=-1),
                         param_grid=dict(max_features=range(8,15)),
                         scoring='neg_mean_absolute_error',
                         cv=ShuffleSplit(1, 0.1))
    
    return model


# ### Plot the individual predictions

# In[ ]:


def evaluation_plot(y, cv_prediction, max_value=None):
    y = np.array(y).ravel()
    cv_prediction = np.array(cv_prediction).ravel()

    result_row = [dataset, pred_property, feature, ml_method, validation_method]
    
    cv_prediction = np.nan_to_num(cv_prediction)
    
    for scorer in ['mean_absolute_error', 'mean_squared_error', 'r2_score']:
        score = getattr(metrics,scorer)(y, cv_prediction)
        if scorer != 'mean_squared_error':
            result_row.append(score)
        else:
            result_row.append(math.sqrt(score))
        print(scorer, score)

    if max_value:
        positive_number = 0
        for i in range(len(cv_prediction)):
            if cv_prediction[i] > max_value[i]:
                positive_number += 1
        result_row.append(positive_number / len(cv_prediction))
    
    with open('{}/results/results.csv'.format(data_folder),'a', newline='') as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerow(result_row)   
    
    file_name = '{}/results/{}_{}_{}_{}_{}'.format(data_folder, dataset, pred_property, feature, 
                                                   ml_method, validation_method)
    
    if max_value:
        data_to_save = {'DFT': list(y), 'ML': list(cv_prediction), 'train_max': max_value}
    else:
        data_to_save = {'DFT': list(y), 'ML': list(cv_prediction)}
    
    pd.DataFrame.from_dict(data_to_save).to_csv(file_name + '.csv', index=False)
    pd.DataFrame.from_dict(data_to_save).to_pickle(file_name + '.pkl')
    
    fig, ax = plt.subplots()

    ax.hist2d(pd.to_numeric(y), cv_prediction, norm=LogNorm(), 
              bins=128, cmap='Blues', alpha=0.9)

    ax.set_xlim(ax.get_ylim())
    ax.set_ylim(ax.get_xlim())

    mae = metrics.mean_absolute_error(y, cv_prediction)
    r2 = metrics.r2_score(y, cv_prediction)
    rmse = math.sqrt(metrics.mean_squared_error(y, cv_prediction))
    
    ax.set_title('{}, {}, {}, {}, {}'.format(dataset, pred_property, feature, ml_method, validation_method))
    ax.text(0.5, 0.1, 'MAE: {:.4f} eV/atom\nRMSE: {:.4f} eV/atom\n$R^2$:  {:.4f}'.format(mae, rmse, r2),
            transform=ax.transAxes,
            bbox={'facecolor': 'w', 'edgecolor': 'k'})

    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')

    ax.set_xlabel('DFT $\Delta H_f$ (eV/atom)')
    ax.set_ylabel('ML $\Delta H_f$ (eV/atom)')

    fig.set_size_inches(5, 5)
    fig.tight_layout()
    fig.savefig(file_name + '.png', dpi=640)

    return mae, rmse, r2


# ### K-fold Cross Validation

# In[ ]:

def holdout(original_model, X, y, shape=None):
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if split == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        split_ratio = 0.8
        split_number = int(split_ratio * len(X))
        X_train = X[:split_number]
        X_test = X[split_number:split_number+100]
        y_train = y[:split_number]
        y_test = y[split_number:split_number+100]
    
    if issubclass(type(original_model), BaseEstimator): 
        model = clone(original_model)
        model.fit(X_train, y_train)
    else:
        model = original_model(shape, l1_reg=False)
        model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1, validation_data=(X_test, y_test))
        model.save_weights('my_model_weights.h5')
        model = original_model(shape, l1_reg=True)
        model.load_weights('my_model_weights.h5')
        model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1, validation_data=(X_test, y_test))


    evaluation_plot(y_test, model.predict(X_test))

def hybrid_train(original_model, X_train, y_train, shape):
    reg_sched = 0.2
    start_epoch = math.floor(epochs * reg_sched)

    model = original_model(shape, l1_reg=False)
    model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0)
    model.save_weights('my_model_weights.h5')
    model = original_model(shape, l1_reg=True)
    model.load_weights('my_model_weights.h5')
    model.fit(X_train, y_train, epochs=30, batch_size=128, verbose=0)
    return model

def cv(original_model, X, y, shape=None, k=k):
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if issubclass(type(original_model), BaseEstimator): 
        cv_prediction = cross_val_predict(original_model, X, y, cv=KFold(k, shuffle=True))
    else:
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # define 10-fold cross validation test harness
        kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
        
        y_random = []
        cv_prediction = []
        for i, (train, test) in enumerate(kfold.split(X, y)):
            sys.stdout.write('{} of {} fold \r'.format(i+1, k))
            sys.stdout.flush()

            # # train model

            if not hybrid:
                model = original_model(shape)
                model.fit(X[train], y[train], epochs=epochs, batch_size=128, verbose=0)
                #validation_data=(X[test], y[test])
            else:
                model = hybrid_train(original_model, X[train], y[train], shape)
    
            y_random.extend(y[test])
            cv_prediction.extend(model.predict(X[test]))
            K.clear_session()
        
        y = y_random
    
    return evaluation_plot(y, cv_prediction)


# ### K-fold Forward/Backward Validation

# In[ ]:


def fcv(original_model, X, y, minimum_ratio=0.1, maximum_ratio=0.95, reverse=False, shape=None, lite=False, k=k):
    if isinstance(X, pd.DataFrame):
        X = X.values
        
    if not reverse:
        arr1inds = y.argsort()
    else:
        arr1inds = y.argsort()[::-1]
    X = X[arr1inds]
    y = y[arr1inds]
    
    if not maximum_ratio:
        maximum_ratio = 1 - 1 / k
    sample_number = len(X)
    fold_sample_number = math.floor(sample_number / k)
    minimum_sample_number = round(minimum_ratio * sample_number)
    maxmum_sample_number = round(maximum_ratio * sample_number)
    
    label = []
    prediction = []
    max_value = []
    
    for split in range(fold_sample_number, sample_number, fold_sample_number if not lite else fold_sample_number*10):
        if split < minimum_sample_number:
            continue
        if split > maxmum_sample_number:
            break
        
#         print("Training 0 to {}, validation on {} to {}".format(split, split, split + fold_sample_number))
        sys.stdout.write("Training end in %s out of %s \r" % (split, sample_number))
        sys.stdout.flush()
        X_train = X[0:split]
        y_train = y[0:split]
        start_sample_number = split + (m - 1) * fold_sample_number
        end_sample_number = split + m * fold_sample_number

        if start_sample_number > sample_number:
            break
        else:
            end_sample_number = min(end_sample_number, sample_number)

        X_val = X[start_sample_number:end_sample_number]
        y_val = y[start_sample_number:end_sample_number]
        
        if issubclass(type(original_model), BaseEstimator): 
            model = clone(original_model)
            model.fit(X_train, y_train)
        else:
            if not hybrid:
                model = original_model(shape)
                model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0)
            else:
                model = hybrid_train(original_model, X_train, y_train, shape)
        y_pred = model.predict(X_val)
    
        label.extend(list(y_val))
        prediction.extend(list(y_pred))
        max_value.extend([y[split-1]] * len(y_val))

        if not issubclass(type(original_model), BaseEstimator):
            K.clear_session()
    
    return evaluation_plot(label, prediction, max_value)

def bcv(model, X, y, k=100, minimum_ratio=0.1, maximum_ratio=0.9, shape=None):
    fcv(model, X, y, k, minimum_ratio, maximum_ratio, reverse=True, shape=shape)

def fbv(model, X, y, training_start=0.1, training_end=0.9, k=None, shape=None):
    if isinstance(X, pd.DataFrame):
        X = X.values

    start = int(training_start*len(y))
    end = int(training_end*len(y))

    X_train = X[start:end]
    y_train = y[start:end]

    X_val_before = X[0:start]
    y_val_before = y[0:start]
    X_val_after = X[end:len(y)]
    y_val_after = y[end:len(y)]

    if issubclass(type(model), BaseEstimator): 
        model = clone(model)
        model.fit(X_train, y_train)
    else:
        model = model(shape)
        model.fit(X_train, y_train, epochs=30, batch_size=128, verbose=0)
    
    y_train_pred = model.predict(X_train)
    y_val_before_pred = model.predict(X_val_before)
    y_val_after_pred = model.predict(X_val_after)

    y_label = np.concatenate((y_val_before, y_train, y_val_after), axis=0)
    y_pred = np.concatenate((y_val_before_pred, y_train_pred, y_val_after_pred), axis=0)

    evaluation_plot(y_label, y_pred)

def iecv(original_model, X, y, minimum_ratio=0.1, maximum_ratio=0.95, reverse=False, shape=None, lite=True):
    cv_result = cv(original_model, X, y, shape, k=5)
    fcv_result = fcv(original_model, X, y, minimum_ratio, maximum_ratio, reverse, shape, lite, k=100)
    print(cv_result[0])
    print(fcv_result[0])
    alpha = 0.5
    beta = 1 - alpha
    print(math.sqrt(alpha * cv_result[0]**2 + beta * fcv_result[0]**2))


# ## 5. Leave one out cross validation/K fold forward step cross validation

# In[ ]:

if __name__ == '__main__':
    main()
