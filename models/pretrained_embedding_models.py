
import sys
import os

from itertools import product
from KGEkeras import DistMult, HolE, TransE, HAKE, ConvE, ComplEx, ConvR, RotatE, pRotatE, ConvKB,CosinE

from kerastuner import RandomSearch, HyperParameters, Objective, Hyperband, BayesianOptimization

from random import choice
from collections import defaultdict

from tensorflow.keras.losses import binary_crossentropy,hinge,mean_squared_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, TerminateOnNaN, ReduceLROnPlateau
from sklearn.metrics.cluster import completeness_score

from tensorflow.keras.optimizers import Adam
import json

from utils import pad
import string
import random

#from tensorflow_addons.callbacks import TimeStopping
SECONDS_PER_TRAIL = 600
SECONDS_TO_TERMINATE = 3600
SEARCH_MAX_EPOCHS = 10
MAX_EPOCHS = 100
MAX_TRIALS = 20

EPSILON = 10e-7

models = {
            'DistMult':DistMult,
            'TransE':TransE,
            'HolE':HolE,
            'ComplEx':ComplEx,
            'HAKE':HAKE,
            'pRotatE':pRotatE,
            'RotatE':RotatE,
            'ConvE':ConvE,
            'ConvKB':ConvKB,
         }


    
import IPython
class ClearTrainingOutput(Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

def build_model(hp):
    
    params = hp.values.copy()
    
    embedding_model = models[params['embedding_model']]
    
    params['e_dim'] = params['dim']
    params['r_dim'] = params['dim']
    
    model = embedding_model(
            **params
            )
    
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), 
                  loss=None, 
                  metrics=None, 
                  experimental_run_tf_function=False)
    
    return model

class MyCallback(Callback):
    """Add relative loss to logs. 
    Relative loss is the loss compared to loss at first epoch.
    """
    def __init__(self, kg, *args, **kwargs):
        #super(Callback, self).__init__(*args, **kwargs)
        self.rl = EPSILON
        self.kg = kg
    
    def on_train_begin(self, logs={}):
        logs = logs or {}
        l = self.model.evaluate(self.kg, batch_size=self.model.get_config()['batch_size'])
        self.rl = max(EPSILON,l)
    
    def on_epoch_end(self, epoch, logs = {}):
        #if not hasattr(self,'rl'):
            #self.rl = logs['loss']
        if not logs is None:
            if np.isnan(logs['loss']) or np.isinf(logs['loss']):
                logs['relative_loss'] = 1
            else:
                logs['relative_loss'] = max(EPSILON,logs['loss'])/self.rl

def optimize_model(model, kg1, kg2):
   
    bs = int(256)
    kg1 = pad(kg1,bs)
    kg2 = pad(kg2,bs)
    kg1 = np.asarray(kg1)
    kg2 = np.asarray(kg2)
    
    embeddings = {}
    
    model_name = model
    
    for kg,name in zip([kg1,kg2],['_chemical','_taxonomy']):
        
        N = len(set([s for s,_,_ in kg]) | set([o for _,_,o in kg]))
        M = len(set([p for _,p,_ in kg]))
            
        hp = HyperParameters()
        hp.Fixed('embedding_model', model_name)
        hp.Fixed('num_entities',value=N)
        hp.Fixed('num_relations',value=M)
        
        lfs = ['pairwize_hinge','pairwize_logistic','pointwize_hinge','pointwize_logistic']
        
        hp.Int('margin',1,10,default=1)

        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        
        if model in ['ConvE','ConvR','ConvKB']:
            batch_size = 128
            hp.Fixed('hidden_dp',0.2)
        else:
            batch_size = bs
            
        hp.Choice('regularization',[0.0,0.01,0.001],default=0.0)
        if model_name in ['TransE','HAKE','pRotatE','RotatE']:
            hp.Int('gamma',0,20,default=0)
            
        hp.Choice('loss_function',lfs)
        hp.Fixed('dp', 0.2)
        hp.Choice('dim', [100,200,400], default=200)
        hp.Choice('negative_samples',[10,100], default=10)
        hp.Fixed('batch_size',batch_size)
        
        tuner = BayesianOptimization(
            build_model,
            hyperparameters=hp,
            objective=Objective('relative_loss','min'),
            max_trials=MAX_TRIALS,
            overwrite=True,
            project_name='tmp/'+''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(11))
            )
    
        tuner.search(kg,
                    epochs=SEARCH_MAX_EPOCHS,
                    batch_size=batch_size,
                    callbacks=[ClearTrainingOutput(), 
                                MyCallback(kg),
                                TerminateOnNaN(),
                                TimeStopping(SECONDS_PER_TRAIL),
                                EarlyStopping('loss',min_delta=1e-5,patience=3)],
                    verbose=1)
                    
        tuner.results_summary()
        
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        model = tuner.hypermodel.build(best_hps)
        out = dict()
        for k in best_hps.values.keys():
            out[k] = best_hps.values[k]
        with open('./pretrained_hp/%s%s_kg.json' % (model_name,name), 'w') as fp:
            json.dump(out, fp)
        
        model.fit(kg,
                epochs=MAX_EPOCHS,
                batch_size=batch_size,
                callbacks=[EarlyStopping('loss',min_delta=1e-5,patience=3),
                           ReduceLROnPlateau('loss',min_delta=1e-5,patience=3)])
        embeddings[name] = model.entity_embedding.get_weights()[0]
        
    return embeddings
                

def main():
    d = './results/pretrained_embeddings/'
    
    kg1 = pd.read_csv('./data/chemicals0.csv')
    kg2 = pd.read_csv('./data/taxonomy0.csv')
    
    kg1 = list(zip(kg1['subject'], kg1['predicate'], kg1['object']))
    kg2 = list(zip(kg2['subject'], kg2['predicate'], kg2['object']))
    
    entities1 = set([s for s, p, o in kg1]) | set([o for s, p, o in kg1])
    relations1 = set([p for s, p, o in kg1])
    entities2 = set([s for s, p, o in kg2]) | set([o for s, p, o in kg2])
    relations2 = set([p for s, p, o in kg2])
    
    me1 = {k:i for i,k in enumerate(entities1)}
    me2 = {k:i for i,k in enumerate(entities2)}
    mr1 = {k:i for i,k in enumerate(relations1)}
    mr2 = {k:i for i,k in enumerate(relations2)}
    kg1 = [(me1[s],mr1[p],me1[o]) for s,p,o in kg1]
    kg2 = [(me2[s],mr2[p],me2[o]) for s,p,o in kg2]
    
    
    best_models = {}
    for model,i in models.items():
        embeddings = optimize_model(model,kg1,kg2)
        
        for k, ent in zip(embeddings,[entities1,entities2]):
            W = embeddings[k]
            f = d+model+k
            np.save(f+'_embeddings.npy', W)
            np.save(f+'_ids.npy',np.asarray(list(zip(ent,range(len(ent))))))
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
