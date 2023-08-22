import torch
import torch
import torch.nn as nn
import pickle

def computeL1Norm(model):

    with torch.no_grad():
        L1 = 0

        L1 = L1 + torch.sum(torch.abs(model[1].weight))
        L1 = L1 + torch.sum(torch.abs(model[1].bias))
        L1 = L1 + torch.sum(torch.abs(model[3].weight))
        L1 = L1 + torch.sum(torch.abs(model[3].bias))
        L1 = L1 + torch.sum(torch.abs(model[5].weight))
        L1 = L1 + torch.sum(torch.abs(model[5].bias))

    return L1

def fixSparseStructure(model, mu):
    
    with torch.no_grad():

        W_1 = torch.zeros_like(model[1].weight)
        W_1[0,0] = mu
        W_1[1,1] = mu
        W_1[2,2] = mu
        W_1[3,3] = mu

        b_1 = torch.zeros_like(model[1].bias)

        W_3 = torch.zeros_like(model[3].weight)
        W_3[0,0] = mu
        W_3[1,1] = mu
        W_3[2,2] = mu
        W_3[3,3] = mu

        b_3 = torch.zeros_like(model[3].bias)

        W_5 = torch.zeros_like(model[5].weight)
        
        b_5 = torch.zeros_like(model[5].bias)

        model[1].weight = nn.Parameter(W_1)
        model[1].bias = nn.Parameter(b_1)
        model[3].weight = nn.Parameter(W_3)
        model[3].bias = nn.Parameter(b_3)
        model[5].weight = nn.Parameter(W_5)
        model[5].bias = nn.Parameter(b_5)

def loadModel(model, path, i):

    with open (path,'rb') as pick:
        Weights_all = pickle.load(pick)

    Weights = Weights_all[i]

    W_1 = Weights[0]
    b_1 = Weights[1]
    W_3 = Weights[2]
    b_3 = Weights[3]
    W_5 = Weights[4]
    b_5 = Weights[5]

    with torch.no_grad():

        model[1].weight = nn.Parameter(W_1)
        model[1].bias = nn.Parameter(b_1)
        model[3].weight = nn.Parameter(W_3)
        model[3].bias = nn.Parameter(b_3)
        model[5].weight = nn.Parameter(W_5)
        model[5].bias = nn.Parameter(b_5)


def get_accuracy(model, X, y):
    """Performs a measure of the correct prediction made

       Args: 
          model: neural network model used
          X, y: Could be either the training dataset or testing dataset 
    """
    correct = 0
    total = 0
    with torch.no_grad():
        #for inputs,labels in zip(X,y):
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy
