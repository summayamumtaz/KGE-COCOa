## all in model

from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense, Input, Dropout
import tensorflow as tf
import kerastuner as kt

from kerastuner import HyperParameters, Objective

from tensorflow.keras.callbacks import  EarlyStopping, TerminateOnNaN, TensorBoard, ReduceLROnPlateau
#from tensorflow_addons.callbacks import TimeStopping
from tensorflow.keras.optimizers import Adam

from KGEkeras import DistMult, HolE, TransE, HAKE, ConvE, ComplEx, ConvR, RotatE, pRotatE, ConvKB

import numpy as np
import pandas as pd

from random import choice

from tensorflow.keras.metrics import AUC, Precision, Recall

from .utils import f1, f2, CVTuner, reset_weights, balance_inputs, prep_data, load_data, create_class_weight, prep_data_v2

from tensorflow.keras.backend import log
import json 
from collections import defaultdict


SECONDS_PER_TRAIL = 600
MAX_TRIALS = 20
MAX_EPOCHS = 1000
PATIENCE = 5
PARAMS = {
        'SECONDS_PER_TRAIL':SECONDS_PER_TRAIL,
        'MAX_TRIALS':MAX_TRIALS,
        'MAX_EPOCHS':MAX_EPOCHS,
        'PATIENCE':PATIENCE
    }

models = {
            'DistMult':DistMult,
            'TransE':TransE,
            'HolE':HolE,
            'ComplEx':ComplEx,
            'RotatE':RotatE,
            'pRotatE':pRotatE,
            'HAKE':HAKE,
            'ConvE':ConvE,
            'ConvR':ConvR,
            'ConvKB':ConvKB,
         }

def build_model(hp):
    
    params = hp.values.copy()
    
    params1 = {k.replace('1',''):params[k] for k in params if not '2' in k}
    params2 = {k.replace('2',''):params[k] for k in params if not '1' in k}
    
    params1['e_dim'],params1['r_dim'] = params1['dim'],params1['dim']
    params2['e_dim'],params2['r_dim'] = params2['dim'],params2['dim']
    
    m1 = models[params1['embedding_model']]
    m2 = models[params2['embedding_model']]
    
    embedding_model1 = m1(**params1)
    embedding_model2 = m2(**params2)
    
    triple1 = Input((3,))
    triple2 = Input((3,))
    ci = Input((1,))
    si = Input((1,))
    conc = Input((1,))
    inputs = [triple1, triple2, ci, si, conc]
        
    _,l1 = embedding_model1(triple1)
    _,l2 = embedding_model2(triple2)
    
    c = embedding_model1.entity_embedding(ci)
    s = embedding_model2.entity_embedding(si)
    c = tf.squeeze(c,axis=1)
    s = tf.squeeze(s,axis=1)
    
    for i,layer_num in enumerate(range(params['branching_num_layers_chemical'])):
        c = Dense(params['branching_units_chemical_'+str(i+1)],activation='relu')(c)
        c = Dropout(0.2)(c)
    
    for i,layer_num in enumerate(range(params['branching_num_layers_species'])):
        s = Dense(params['branching_units_species_'+str(i+1)],activation='relu')(s)
        s = Dropout(0.2)(s)
    
    for i,layer_num in enumerate(range(hp.Int('branching_num_layers_conc',0,3,default=1))):
        conc = Dense(params['branching_units_conc_'+str(i+1)],activation='relu')(conc)
        conc = Dropout(0.2)(conc)
    
    x = Concatenate(axis=-1)([c,s,conc])
    
    for i,layer_num in enumerate(range(hp.Int('num_layers',0,3,default=1))):
        x = Dense(params['units_'+str(i+1)],activation='relu')(x)
        x = Dropout(0.2)(x)
        
    x = Dense(params['output_dim'],activation='sigmoid',name='output_1')(x)
    
    model = Model(inputs=inputs, outputs=[x])
    model.add_loss(params1['loss_weight']*l1/2 + params2['loss_weight']*l2/2)
    
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss={'output_1':'binary_crossentropy'},
                  loss_weights={'output_1':params['classification_loss_weight']},
                  metrics=['acc', f1, f2, Precision(), Recall(), AUC()])
    
    return model

        
def fit_sim_model(X_train, X_test, y_train, y_test, model1, model2, results_file='results.csv', embedding_file='sim_embeddings', num_runs=1, hp_file1=None, hp_file2=None, hp_pred_file=None, params=None):
    params = params or PARAMS
    
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
    
    output_dim = 1
    
    X_train,y_train = np.asarray([(me1[a],me2[b],float(x)) for a,b,x in X_train if a in entities1 and b in entities2]), np.asarray([float(x) for x,a in zip(y_train,X_train) if a[0] in entities1 and a[1] in entities2])
    
    X_test, y_test = np.asarray([(me1[a],me2[b],float(x)) for a,b,x in X_test if a in entities1 and b in entities2]), np.asarray([float(x) for x,a in zip(y_test, X_test) if a[0] in entities1 and a[1] in entities2])
        
    scores = []    
    k_best_predictions = []
    
    hp = HyperParameters()
    
    kg_lengths = list(map(len,[kg1,kg2]))
    output_lengths = len(X_train)
    
    hp.Fixed('num_entities1',len(entities1))
    hp.Fixed('num_entities2',len(entities2))
    hp.Fixed('num_relations1',len(relations1))
    hp.Fixed('num_relations2',len(relations2))
    
    hp.Fixed('embedding_model1',model1)
    hp.Fixed('embedding_model2',model2)
    hp.Fixed('output_dim',output_dim)
    
    bs = 1024
    
    if hp_file1 and hp_file2:
        for i,hp_file in enumerate([hp_file1,hp_file2]):
            with open(hp_file, 'r') as fp:
                data = json.load(fp)
                for k in data:
                    hp.Fixed(k+str(i+1),data[k])
                    if k == 'batch_size':
                        bs = min(bs,data[k])
    else:
        for i,m in zip(['1','2'],[model1,model2]):
            hp.Choice('dim'+i,[100,200,400],default=200)
            hp.Choice('negative_samples'+i,[10,100],default=10)
            if m in ['ConvE','ConvR','ConvKB']:
                bs = 128
            hp.Choice('loss_function'+i,['pairwize_hinge','pairwize_logistic','pointwize_hinge','pointwize_logistic'],default='pairwize_hinge')
            w = kg_lengths[int(i)-1]/max(kg_lengths)
    
    if hp_pred_file:
        with open(hp_pred_file, 'r') as fp:
            data = json.load(fp)
            for k in data:
                hp.Fixed(k,data[k])
    else:
        MAX_LAYERS = 3
        hp.Int('branching_num_layers_chemical',0,MAX_LAYERS,default=1)
        hp.Int('branching_num_layers_species',0,MAX_LAYERS,default=1)
        hp.Int('branching_num_layers_conc',0,MAX_LAYERS,default=1)
        hp.Int('num_layers1',0,3,default=1)
        for i in range(MAX_LAYERS+1):
            hp.Choice('branching_units_chemical_'+str(i+1),[32,128,512], default=128)
            hp.Choice('branching_units_species_'+str(i+1),[32,128,512], default=128)
            hp.Choice('branching_units_conc_'+str(i+1),[32,128,512], default=128)
            hp.Choice('units_'+str(i+1),[32,128,512], default=128)
            
    
    # Since inputs are oversampled, we must reduce the weight of losses accordingly. 
    w = output_lengths/max(kg_lengths)
    hp.Float('loss_weight1', w, 5*w, sampling='log')
    hp.Float('loss_weight2', w, 5*w, sampling='log')
    hp.Float('classification_loss_weight', w, 5*w, sampling='log')
    hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    hp.Fixed('batch_size',bs)
    
    m = max(map(len,[kg1,kg2,X_train])) + (bs - max(map(len,[kg1,kg2,X_train])) % bs)
    Xtr, ytr = prep_data_v2(kg1,kg2,X_train,y_train,max_length=m)
    Xte, yte = prep_data_v2(kg1,kg2,X_test,y_test,test=True,max_length=max(bs,len(y_test)))
    
    tuner = CVTuner(
            hypermodel=build_model,
            oracle=kt.oracles.BayesianOptimization(
                hyperparameters=hp,
                objective=Objective('val_auc','max'),
                max_trials=params['MAX_TRIALS']),
            overwrite=True,
            project_name='tmp/'+''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(11))
            )
    
    tuner.search(Xtr,ytr,
        epochs=params['SEARCH_MAX_EPOCHS'],
        batch_size=bs,
        callbacks=[EarlyStopping('loss',mode='min',patience=params['PATIENCE'])],
        kfolds=params['NUM_FOLDS'],
        class_weight = params['cw']
        )
    
    results = []
    prediction = []
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    model = tuner.hypermodel.build(best_hps)
    
    out = dict()
    for k in best_hps.values.keys():
        out[k] = best_hps.values[k]
    with open('./sim_hp/%s.json' % hp_pred_file.split('/')[-1].split('_')[0], 'w') as fp:
        json.dump(out, fp)
    
    for _ in range(num_runs):
        reset_weights(model)
        model.fit(Xtr,ytr,
            epochs=params['MAX_EPOCHS'],
            batch_size=bs,
            verbose=2,
            class_weight = params['cw'],
            callbacks=[EarlyStopping('loss',mode='min',patience=params['PATIENCE'])]
            )
        r = model.evaluate(Xte,yte,verbose=0,batch_size=bs)
        results.append(r)
        
    W1 = model.get_layer('embedding').get_weights()[0]
    W2 = model.get_layer('embedding_2').get_weights()[0]
    np.save(embedding_file+'_chemical_embeddings.npy', W1)
    np.save(embedding_file+'_chemical_ids.npy',np.asarray(zip(entities1,range(len(entities1)))))
    np.save(embedding_file+'_taxonomy_embeddings.npy', W2)
    np.save(embedding_file+'_taxonomy_ids.npy',np.asarray(zip(entities2,range(len(entities2)))))
    
    var = np.var(np.asarray(results),axis=0)
    results = np.mean(np.asarray(results),axis=0)
    
    df = pd.DataFrame(data={'metric':model.metrics_names,'value':list(results), 'variance':list(var)})
    df.to_csv(results_file)

