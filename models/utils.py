#utils 

import matplotlib.pyplot as plt
import numpy as np
import kerastuner
import keras 
from keras.losses import cosine_proximity
import keras.backend as K
from random import choices, choice
import pandas as pd

#import kerastuner
import numpy as np
from sklearn import model_selection
from itertools import product
import tensorflow as tf
import math

def prep_data_v2(kg1,kg2,X,y,test=False, max_length=0):
    kg1,kg2,X,y = list(kg1),list(kg2),list(X),list(y)
    m = max(map(len,[kg1,kg2,X,y]))
    m = max(m,max_length)
    if test:
        m = max(len(X),max_length)
        kg1 = kg1[:m]
        kg2 = kg2[:m]
    while min(map(len,[kg1,kg2,X,y])) < m:
        if len(kg1) < m:
            kg1.append(choice(kg1))
        if len(kg2) < m:
            kg2.append(choice(kg2))
        if len(X) < m:
            idx = choice(range(len(X)))
            X.append(X[idx])
            y.append(y[idx])
            
    c,s,conc = zip(*X)
    return [np.asarray(kg1),np.asarray(kg2),np.asarray(c),np.asarray(s),np.asarray(conc)],np.asarray(y)

def pad(kg,bs):
    while len(kg) % bs != 0:
        kg.append(choice(kg))
    return kg

        
class CVTuner(kerastuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, batch_size=32, epochs=1,callbacks=None, kfolds=5, class_weight=None):
        cv = model_selection.KFold(kfolds,shuffle=True,random_state=42)
        val_losses = []
        for train_indices, test_indices in cv.split(x[0]):
            x_train1, x_test1 = x[0][train_indices], x[0][test_indices]
            x_train2, x_test2 = x[1][train_indices], x[1][test_indices]
            
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit([x_train1, x_train2], y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2)
            val_losses.append(model.evaluate([x_test1,x_test2], y_test, verbose=0, batch_size=batch_size))
            m = np.mean(val_losses,axis=0)
            d = dict([('val_'+mn,vl) for mn,vl in zip(model.metrics_names,m)])
            self.oracle.update_trial(trial.trial_id, d)
            self.save_model(trial.trial_id, model)

def reset_weights(model):
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
    for k, initializer in layer.__dict__.items():
        if "initializer" not in k:
            continue
        # find the corresponding variable
        var = getattr(layer, k.replace("_initializer", ""))
        var.assign(initializer(var.shape, var.dtype))
        
def create_class_weight(labels_dict,mu=0.15):
    total = np.sum([v for k,v in labels_dict.items()])
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/labels_dict[key])
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

def train_test_split_custom(X, Y, test_size=0.20, random_state=42, sampling = 'all'):
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    entities1, entities2, _ = map(list, zip(*X))
    entities = [set(entities1), set(entities2)]
    
    if sampling == 'none':
        return X_train, X_test, y_train, y_test
    
    X_train_tmp = []
    X_test_tmp = []
    y_train_tmp = []
    y_test_tmp = []
    
    if sampling == 'chemical' or sampling == 'species':
        if sampling == 'chemical': k = 0
        if sampling == 'species': k = 1
        tmp_train, tmp_test = model_selection.train_test_split(list(entities[k]),test_size=test_size, random_state=random_state)

        for x,y in zip(X_train, y_train):
            if x[k] in tmp_train:
                X_train_tmp.append(x)
                y_train_tmp.append(y)
            else:
                X_test_tmp.append(x)
                y_test_tmp.append(y)
                
        for x,y in zip(X_test, y_test):
            if x[k] in tmp_test:
                X_test_tmp.append(x)
                y_test_tmp.append(y)
            else:
                X_train_tmp.append(x)
                y_train_tmp.append(y)
    
    if sampling == 'both':
        X_train_tmp = []
        X_test_tmp = []
        y_train_tmp = []
        y_test_tmp = []
        # approx 0.80/0.20 split
        p1 = 1-(1-test_size)**2
        p2 = 1-(1-test_size)**2
        
        A1, A2 = model_selection.train_test_split(list(entities[0]),test_size=p1, random_state=random_state)
        B1, B2 = model_selection.train_test_split(list(entities[1]),test_size=p2, random_state=random_state)
    
        tmp_train_pairs = set(product(A1,B1))
        tmp_test_pairs = set(product(A2,B2))
        
        for x,y in zip(X,Y):
            if (x[0],x[1]) in tmp_train_pairs:
                X_train_tmp.append(x)
                y_train_tmp.append(y)
            if (x[0],x[1]) in tmp_test_pairs:
                X_test_tmp.append(x)
                y_test_tmp.append(y)

    X_train, X_test, y_train, y_test = list(map(np.asarray, [X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp]))
        
    # for sanity
    if sampling == 'chemical':
        assert len(set((X_train[:,0])).intersection(set(X_test[:,0]))) == 0
            
    if sampling == 'species':
        assert len(set((X_train[:,1])).intersection(set(X_test[:,1]))) == 0
    
    if sampling == 'both':
        assert len(set((X_train[:,0])).intersection(set(X_test[:,0]))) == 0 and len(set((X_train[:,1])).intersection(set(X_test[:,1]))) == 0
    
    return X_train, X_test, y_train, y_test
    

def load_data(filename):
    df = pd.read_csv(filename).dropna()
    f = lambda x: 1 if x == 'https://cfpub.epa.gov/ecotox/effect/MOR' else 0
    df['effect'] = df['effect'].apply(f)
    df = df.groupby(['chemical','species'],as_index=False).median()
    df1 = df[df['effect'] == 1]
    df2 = df[df['effect'] == 0]
    df = pd.concat((df1,df2))
    df['concentration'] = (df['concentration']-df['concentration'].min())/(df['concentration'].max()-df['concentration'].min())
    X, y = list(zip(df['chemical'],df['species'],df['concentration'])),list(df['effect'])
    return X,y 

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p

def f1(y_true, y_pred):
    beta = 1
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2) * (p*r)/(beta**2*p+r + K.epsilon())

def f2(y_true, y_pred):
    beta = 2
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1+beta**2) * (p*r)/(beta**2*p+r + K.epsilon())

class TrainingPlot(keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('output/Epoch-{}.png'.format(epoch))
            plt.close()

def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels = (1-factor)*labels + factor/labels.shape[1]

    # returned the smoothed labels
    return labels

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, kg1, kg2, X, y, batch_size=32, shuffle=True, smoothing=False):
        'Initialization'
        self.batch_size = batch_size
        self.smoothing = smoothing
        
        self.kg1 = kg1
        self.kg2 = kg2
        #self.N1 = len(set([a for a,b,c in kg1]) | set([c for a,b,c in kg1]))
        #self.N2 = len(set([a for a,b,c in kg2]) | set([c for a,b,c in kg2]))
        #self.use_kg = True
            
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.y)//self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.asarray(self.X[idx])
        y = np.asarray(self.y[idx])
        
        tmp1 = choices(self.kg1, k = len(idx))
        tmp2 = choices(self.kg2, k = len(idx))
        
        inputs, outputs = prep_data([tmp1,tmp2,X],[y])
            
        if self.smoothing:
            outputs = [smooth_labels(y) for y in outputs]
        
        if self.shuffle:
            idx = np.arange(len(outputs[0]))
            np.random.shuffle(idx)
            inputs = [a[idx] for a in inputs]
            outputs = [a[idx] for a in outputs]
        
        return inputs, outputs

def balance_inputs(inputs,inputs_labels):
    #inputs = [list(i) for i in inputs]
    #inputs_labels = [list(i) for i in inputs_labels]
    
    input_copy = [list(i) for i in inputs]
    input_labels_copy = [list(i) for i in inputs_labels]
    
    l = lengths(input_labels_copy)
    while len(set(l)) > 1:
        list_idx = np.argmin(l)
        idx = choice(range(len(inputs[list_idx])))
        input_copy[list_idx].append(inputs[list_idx][idx])
        input_labels_copy[list_idx].append(inputs_labels[list_idx][idx])
        l = lengths(input_labels_copy)

    return input_copy, input_labels_copy

def generate_negative(kg, N, negative=2, check_kg=False):
    # false triples:
    true_kg = kg.copy()
    kg = []
    kgl = []
    for s, p, o in true_kg:
        kg.append((s, p, o))
        kgl.append(1)
        for _ in range(negative):
            t = (choice(range(N)), p, choice(range(N)))
            
            if check_kg:
                if not t in true_kg:
                    kg.append(t)
                    kgl.append(0)
            else:
                kg.append(t)
                kgl.append(0)
    
    return kg, kgl

def prep_data(inputs,outputs):
    triples1,triples2,x = inputs
    
    x1,x2 = zip(*x)
    inputs = [np.asarray(a) for a in [triples1,triples2,x1,x2]]
    inputs[-1] = inputs[-1].reshape((-1,1))
    inputs[-2] = inputs[-2].reshape((-1,1))
    outputs = [np.asarray(a).reshape((-1,1)) for a in outputs]
    return inputs, outputs

def lengths(inputs):
    return [len(i) for i in inputs]

def joint_cosine_loss(x):
    """
    x : dot(chemical,species)
    """
    def func(y_true, y_pred):
        return K.reduce_sum(y_true*x + (y_true-1)*x)
    return func


def prep_data(kg1,kg2,data,labels,test=False):
    
    if not (kg1 and kg2):
        c,s = zip(*data)
        return [np.asarray(c).reshape(-1,),np.asarray(s).reshape(-1,)], [np.asarray(labels).reshape(-1,)]
    
    if test:
        kg1 = kg1[:min(len(kg1),len(data))]
        kg2 = kg2[:min(len(kg2),len(data))]
        
    inputs, outputs = balance_inputs([kg1,kg2,data],[np.zeros((len(kg1))),np.zeros((len(kg2))),labels])
    
    outputs = [outputs[-1]]
    
    kg1,kg2,data = inputs
    c,s = zip(*data)
    inputs = [np.asarray(kg1), np.asarray(kg2), np.asarray(c).reshape((-1,)),np.asarray(s).reshape((-1,))]
    outputs = [np.asarray(a).reshape((-1,)) for a in outputs]
    return inputs, outputs










