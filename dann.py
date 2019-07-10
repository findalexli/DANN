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
#Define Batch Size 
batch_size = 200000

# Dataset 
train_loader = torch.utils.data.TensorDataset(X_train_tensor,y_train_tensor)
    
test_dataset = torch.utils.data.TensorDataset(x_tensor_va,y_tensor_va)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                          shuffle=False)

#Defining Dataset from dataloader
test_dataset = torch.utils.data.TensorDataset(x_tensor_va,y_tensor_va)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Constructing the neuralnet from scratch. We added 3 more layers (6 layers in total)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size) 
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size) 
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, hidden_size) 
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_size, num_classes) 
        self.dropout = nn.Dropout(0.1)
        
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.dropout(out)
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.dropout(out)
        out = self.fc6(out)
        
        return out
#Check if cpu is availble, for user's information
torch.cuda.is_available()

# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading the model 

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Define the number of epochs
num_epochs = 10
# Display the number of epoches
print ('batch_size = ' + batch_size)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):  
        
        data = data.cuda()
        labels = labels.cuda()
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
# Let us save the model immediately 
torch.save(model, 'savedodel.ckpt')


torch.load('model2.ckpt')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model2.ckpt')

