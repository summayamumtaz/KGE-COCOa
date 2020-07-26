## Use pretrained embeddings to train a auto-sklearn classisifier.

import numpy as np
import pandas as pd
import tensorflow as tf
import kerastuner as kt

from kerastuner import HyperParameters, Objective

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate
#from tensorflow_addons.callbacks import TimeStopping
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy, Metric
from tensorflow.keras.optimizers import Adam

from .utils import f1, f2, CVTuner, reset_weights, create_class_weight

import json
import random
import string

MAX_EPOCHS = 1000
SECONDS_PER_TRAIL = 600
MAX_TRIALS = 20
PATIENCE = 5
PARAMS = {
        'SECONDS_PER_TRAIL':SECONDS_PER_TRAIL,
        'MAX_TRIALS':MAX_TRIALS,
        'MAX_EPOCHS':MAX_EPOCHS,
        'PATIENCE':PATIENCE
    }


         
def build_model(hp, input_dim1, input_dim2,output_dim, first_layer=False):
    
    params = hp.values.copy()
    
    ci = Input((input_dim1,))
    si = Input((input_dim2,))
    s=si
    
    for i in range(hp.Int('num_layers', 0, 2)):
        s = Dense(hp.Choice('branching_units'+str(i+1),units,default=units[0]),activation='relu')(s)
        s = Dropout(0.2)(s)
        
    x = Concatenate(axis=-1)([ci,s])
    
    x1 = Dense(hp.Choice('units_'+str(1),[16,8], default=16), activation='relu')(x)
   
        
    x = Dense(output_dim,activation='softmax',name='output_1')(x1)
    
    model = Model(inputs=[ci,si],outputs=[x])
       
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'output_1':'categorical_crossentropy'},
                  metrics=['acc', f1, f2, Precision(), Recall(), AUC()])
    
    return model

def tune(X_train, X_test, y_train, y_test, 
         hypermodel,
         hp,
         params,
         num_runs,
         results_file,
         hp_file):
    
    tuner = CVTuner(
        hypermodel=hypermodel,
        oracle=kt.oracles.BayesianOptimization(
            hyperparameters=hp,
            objective=Objective('val_auc','max'),
            max_trials=params['MAX_TRIALS']),
        overwrite=True,
        project_name='tmp/'+''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(11))
        )

    tuner.search(X_train,y_train,
        epochs=params['MAX_EPOCHS'],
        batch_size=1024,
        callbacks=[EarlyStopping('loss',mode='min',patience=params['PATIENCE']),
                   ReduceLROnPlateau('loss',mode='min',patience=params['PATIENCE'])],
        kfolds=params['NUM_FOLDS']
        )

    results = []
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    model = tuner.hypermodel.build(best_hps)
    for _ in range(num_runs):
        model.fit(X_train, y_train,
                epochs=params['MAX_EPOCHS'],
                batch_size=1024,
                callbacks=[EarlyStopping('loss',mode='min',patience=params['PATIENCE']),
                           ReduceLROnPlateau('loss',mode='min',patience=params['PATIENCE'])],
                verbose=0)
        r = model.evaluate(X_test,y_test,verbose=0)
        results.append(r)
        reset_weights(model)
            
    var = np.var(np.asarray(results),axis=0)
    results = np.mean(np.asarray(results),axis=0)
    
    df = pd.DataFrame(data={'metric':model.metrics_names,'value':list(results), 'variance':list(var)})
    df.to_csv(results_file)
    
    out = dict()
    for k in best_hps.values.keys():
        out[k] = best_hps.values[k]
    with open(hp_file, 'w') as fp:
        json.dump(out, fp)

class PriorModel:
    def __init__(self):
        pass
    def fit(self,X,y):
        u,uw = np.unique(y,return_counts=True)
        self.lookup = uw/sum(uw)
    
    def predict(self,X):
        return np.asarray([np.argmax(self.lookup) for _ in range(len(X))])

def fit_onehot(X_train, X_test, y_train, y_test, output_dim, results_file='results.csv',hp_file='hp.json',num_runs=1, params=None):
    #one hot
    params = params or PARAMS
    
     
    hp = HyperParameters()
    
    dim1 = X_train[0].shape[1]
    dim2 = X_train[1].shape[1]
    out_dim = output_dim
    bm = lambda x: build_model(hp, dim1, dim2, out_dim)
    tune(X_train, X_test, y_train, y_test, 
         bm,
         hp,
         params,
         num_runs,
         results_file,
         hp_file)


      

def fit_hier_embeddings(X_train, X_test, y_train, y_test, out_dim, chemical_embedding_files,  taxonomy_embedding_files,results_file='results.csv',hp_file='hp.json', num_runs=1, params=None):
    params = params or PARAMS

    hp = HyperParameters()
    
    bm = lambda x: build_model(x,len(X_train[0][0]),len(X_train[1][0]),out_dim )
    tune(X_train, X_test, y_train, y_test, 
         bm,
         hp,
         params,
         num_runs,
         results_file,
         hp_file)
    
