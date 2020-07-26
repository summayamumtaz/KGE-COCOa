

from models.utils import load_data, train_test_split_custom

#from models.sim_embedding_models import fit_sim_model

from models.pretrained_auto_keras import fit_onehot,  fit_hier_embeddings
from keras.utils import to_categorical
from sklearn import preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
import kerastuner as kt
import kerastuner
from kerastuner.tuners import RandomSearch, BayesianOptimization
from kerastuner import HyperParameters, Objective
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate
#from tensorflow_addons.callbacks import TimeStopping
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy, Metric
from tensorflow.keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.utils import class_weight
import glob




DATA_FILE = './data/cocoa_data.csv'
NUM_RUNS = 1
SECONDS_PER_TRAIL = 600
MAX_TRIALS = 20
MAX_EPOCHS = 100
SEARCH_MAX_EPOCHS = 10
PATIENCE = 10
NUM_FOLDS = 5

PARAMS = {
        'SECONDS_PER_TRAIL':SECONDS_PER_TRAIL,
        'MAX_TRIALS':MAX_TRIALS,
        'MAX_EPOCHS':MAX_EPOCHS,
        'PATIENCE':PATIENCE,
        'SEARCH_MAX_EPOCHS':SEARCH_MAX_EPOCHS,
        'NUM_FOLDS':NUM_FOLDS
    }



def train_test_split_onehot(X, y, output_dim,test_size=0.20):
    X_onehot = pd.get_dummies(X['BeanOrigin'], drop_first=True)
    X = pd.concat( [X[['Rating', 'CocoaPercent']],X_onehot], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    y_train = to_categorical(y_train, output_dim)
    y_test = to_categorical(y_test, output_dim)
    
    X_train1= X_train[['Rating', 'CocoaPercent']].values
    X_train2 = X_train.drop([ 'Rating', 'CocoaPercent'],axis=1).values
    X_test1= X_test[['Rating', 'CocoaPercent']].values
    X_test2 = X_test.drop(['Rating', 'CocoaPercent'],axis=1).values

    X_train = [ X_train1,  X_train2]
    X_test  =  [X_test1, X_test2]
    return X_train, X_test,y_train, y_test
  
    
    
def train_test_split_hier(X, y,output_dim, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    y_train = to_categorical(y_train, output_dim)
    y_test = to_categorical(y_test, output_dim)
    
    X_train1= X_train[['Rating', 'CocoaPercent']].values
    X_train2 = X_train.drop([ 'Rating', 'CocoaPercent'],axis=1).values
    X_test1= X_test[['Rating', 'CocoaPercent']].values
    X_test2 = X_test.drop(['Rating', 'CocoaPercent'],axis=1).values

    X_train = [ X_train1,  X_train2]
    X_test  =  [X_test1, X_test2]
    return X_train, X_test,y_train, y_test
  

def main():
    
    data = pd.read_csv(DATA_FILE)
    X_onehot = data[['BeanOrigin','Rating', 'CocoaPercent' ]]
    y = data['BeanType'].values
    output_dim = data['BeanType'].nunique()
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    
    #fit_one_hot(X,y,'result','cocoa_onehot')
    
    X_hier= data.drop(['BeanType', 'BeanOrigin'], axis=1)
  
    print('Prior',np.unique(y,return_counts=True)[1]/sum(np.unique(y,return_counts=True)[1]))
    
        
    X_train, X_test, y_train, y_test = train_test_split_onehot(X_onehot, y, output_dim,test_size=0.20)
        
        
    # test one-hot model
    fit_onehot(X_train, X_test, y_train, y_test, output_dim,
                results_file='./results/%s_one_hot.csv' , 
                hp_file = './pred_hp/%s_one_hot.csv' ,
                num_runs = NUM_RUNS,
                params=PARAMS)
        
    # test hier embeddings 
    X_train, X_test, y_train, y_test = train_test_split_hier(X_hier, y,output_dim, test_size=0.20)
    fit_hier_embeddings(X_train, X_test, y_train, y_test,output_dim,
                            results_file='./results/%s_hierarchy_embedding.csv' ,
                            hp_file='./pred_hp/%s_hierarchy_embedding.csv' ,
                            num_runs = NUM_RUNS,
                            params=PARAMS)
        
 
          
    
if __name__ == '__main__':
    main()
    
    
    
