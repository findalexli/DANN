#!/usr/bin/env python
from argparse import ArgumentParser
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os
import numpy
import scipy
import pylab
import torch
import torch.nn as nn
    
    
input_size = 949
hidden_size = 1000
num_classes = 2
num_epochs = 10
batch_size = 1000
learning_rate = 0.01

if __name__ == "__main__":
    cadd_dir = '/home/alex'
    ClinVar_ESP_dir = '/home/alex'
  
    print('Load Data')
    X_tr = numpy.load(os.path.join(cadd_dir, 'training.X.npz'))
    X_tr = scipy.sparse.csr_matrix((X_tr['data'], X_tr['indices'], X_tr['indptr']), shape=X_tr['shape'])
    y_tr = numpy.load(os.path.join(cadd_dir, 'training.y.npy'))
    
    X_va = numpy.load(os.path.join(cadd_dir, 'validation.X.npz'))
    X_va = scipy.sparse.csr_matrix((X_va['data'], X_va['indices'], X_va['indptr']), shape=X_va['shape'])
    y_va = numpy.load(os.path.join(cadd_dir, 'validation.y.npy')) 
    
    X_te = numpy.load(os.path.join(cadd_dir, 'testing.X.npz'))
    X_te = scipy.sparse.csr_matrix((X_te['data'], X_te['indices'], X_te['indptr']), shape=X_te['shape'])
    y_te = numpy.load(os.path.join(cadd_dir, 'testing.y.npy'))
    
    X_ClinVar_ESP = numpy.load(os.path.join(ClinVar_ESP_dir, 'ClinVar_ESP.X.npz'))  
    X_ClinVar_ESP = scipy.sparse.csr_matrix((X_ClinVar_ESP['data'], X_ClinVar_ESP['indices'], X_ClinVar_ESP['indptr']), shape=X_ClinVar_ESP['shape'])
    y_ClinVar_ESP = numpy.load(os.path.join(ClinVar_ESP_dir, 'ClinVar_ESP.y.npy'))




x_dense_va = scipy.sparse.csc_matrix.todense(X_va)
x_tensor_va = torch.from_numpy(x_dense_va)

x_tensor_va, y_tensor_va

#need to import data 
import torch.utils.data

# Dataset 
#train_loader = torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
    
test_dataset = torch.utils.data.TensorDataset(x_tensor_va,y_tensor_va)
# Data loader
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                          shuffle=False)

y_tensor_va = torch.from_numpy(y_va)
