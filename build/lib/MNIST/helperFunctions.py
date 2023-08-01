import torch
import numpy as np

import torch
import torch.nn as nn

import os
import pickle

def getActivationStructure(model):

    # activation structure stored in NN
    S = []

    with torch.no_grad():

        S.append(torch.abs(torch.sign(model[1].weight)))
        S.append(torch.abs(torch.sign(model[1].bias)))
        S.append(torch.abs(torch.sign(model[3].weight)))
        S.append(torch.abs(torch.sign(model[3].bias)))
        S.append(torch.abs(torch.sign(model[5].weight)))
        S.append(torch.abs(torch.sign(model[5].bias)))

    return S

def activationStructureIsEqual(S, T):

    bool = True

    if torch.min(S[0] == T[0]) == False:
        bool = False
    
    if torch.min(S[1] == T[1]) == False:
        bool = False

    if torch.min(S[2] == T[2]) == False:
        bool = False

    if torch.min(S[3] == T[3]) == False:
        bool = False

    if torch.min(S[4] == T[4]) == False:
        bool = False

    if torch.min(S[5] == T[5]) == False:
        bool = False

    return bool

def createLatexTikzFromActivationStructure(S, filename):
    
    with open(filename, 'w') as f:

        f.write('\\begin{center}\n')
        f.write('\\begin{tikzpicture}[x=1cm, y=1cm, >=stealth]\n\n')

        f.write('    % nodes input layer\n\n')
        
        f.write('    \\node [every neuron/.try, neuron 1/.try] (input-1) at (4,2-1) {};\n')
        f.write('    \\node [every neuron/.try, neuron 2/.try] (input-2) at (4,2-2) {};\n')
        f.write('    \\node [every neuron/.try, neuron 3/.try] (input-3) at (4,2-3) {};\n')
        f.write('    \\node [every neuron/.try, neuron 4/.try] (input-4) at (4,2-4) {};\n\n')

        f.write('    % nodes hidden layer 1\n\n')

        for j in range(1,5):
            if S[1][j-1] == 0:
                f.write('    \\node [every neuron/.try, neuron ' + str(j) + '/.try] (hidden1-' + str(j) +') at (6.5,2-'+str(j)+') {};\n')
            else: 
                f.write('    \\node [filled neuron/.try, neuron ' + str(j) + '/.try] (hidden1-' + str(j) +') at (6.5,2-'+str(j)+') {};\n')
        f.write('\n')

        f.write('    % nodes hidden layer 2\n\n')

        for j in range(1,5):
            if S[3][j-1] == 0:
                f.write('    \\node [every neuron/.try, neuron ' + str(j) + '/.try] (hidden2-' + str(j) +') at (9,2-'+str(j)+') {};\n')
            else: 
                f.write('    \\node [filled neuron/.try, neuron ' + str(j) + '/.try] (hidden2-' + str(j) +') at (9,2-'+str(j)+') {};\n')
        f.write('\n')

        f.write('    % nodes output layer\n\n')

        for j in range(1,4):
            if S[5][j-1] == 0:
                f.write('    \\node [every neuron/.try, neuron ' + str(j) + '/.try] (output-' + str(j) +') at (11.5,2-'+str(j)+'*1.25) {};\n')
            else: 
                f.write('    \\node [filled neuron/.try, neuron ' + str(j) + '/.try] (output-' + str(j) +') at (11.5,2-'+str(j)+'*1.25) {};\n')
        f.write('\n')

            
        f.write('    % in\n\n')

        f.write('    \\foreach \l [count=\i] in {1,2,3,4}\n')
        f.write('        \draw [<-] (input-\i) -- ++(-1,0)\n')
        f.write('        node [above, midway] {$I_\l$};\n\n')

        f.write('    % out\n\n')
            
        f.write('    \\foreach \i in {1,2,3}\n')
        f.write('        \draw [->] (output-\i) -- ++(1,0)\n')
        f.write('        node [above, midway] {$O_\i$};\n\n')

        f.write('    % W^1\n\n')

        for i in range(1,5):
            for j in range(1,5):
                if S[0][i-1, j-1] == 0:
                    f.write('    \draw [gray,->] (input-'+str(j)+') -- (hidden1-'+str(i)+');\n')
                else:
                    f.write('    \draw [red,->] (input-'+str(j)+') -- (hidden1-'+str(i)+');\n')
            f.write('\n')

        f.write('    % W^3\n\n')

        for i in range(1,5):
            for j in range(1,5):
                if S[2][i-1, j-1] == 0:
                    f.write('    \draw [gray,->] (hidden1-'+str(j)+') -- (hidden2-'+str(i)+');\n')
                else:
                    f.write('    \draw [red,->] (hidden1-'+str(j)+') -- (hidden2-'+str(i)+');\n')
            f.write('\n')

        f.write('    % W^5\n\n')

        for i in range(1,4):
            for j in range(1,5):
                if S[4][i-1, j-1] == 0:
                    f.write('    \draw [gray,->] (hidden2-'+str(j)+') -- (output-'+str(i)+');\n')
                else:
                    f.write('    \draw [red,->] (hidden2-'+str(j)+') -- (output-'+str(i)+');\n')
            f.write('\n')

        f.write('    % captions\n\n')
            
        f.write('    \\node [align=center, above] at (4,2) {Input\\ layer};\n')
        f.write('    \\node [align=center, above] at (7.75,2) {Hidden\\ layers};\n')
        f.write('    \\node [align=center, above] at (11,2) {Output\\ layer};\n\n')
        
        f.write('\end{tikzpicture}\n')
        f.write('\end{center}')

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

def computeL0Norm(S):

    with torch.no_grad():
        L0 = 0

        L0 = L0 + torch.sum(S[0])
        L0 = L0 + torch.sum(S[1])
        L0 = L0 + torch.sum(S[2])
        L0 = L0 + torch.sum(S[3])
        L0 = L0 + torch.sum(S[4])
        L0 = L0 + torch.sum(S[5])

    return L0

def fixSparseStructure(model):
    
    with torch.no_grad():

        W_1 = torch.zeros_like(model[1].weight)
        W_1[0,0],W_1[1,1],W_1[2,2],W_1[3,3],W_1[4,4] = 1,1,1,1,1
        W_1[5,5],W_1[6,6],W_1[7,7],W_1[8,8],W_1[9,9] = 1,1,1,1,1
        W_1[10,10],W_1[11,11],W_1[12,12],W_1[13,13],W_1[14,14] = 1,1,1,1,1
        W_1[15,15],W_1[16,16],W_1[17,17],W_1[18,18],W_1[19,19] = 1,1,1,1,1

        b_1 = torch.zeros_like(model[1].bias)

        W_3 = torch.zeros_like(model[3].weight)
        W_3[0,0],W_3[1,1],W_3[2,2],W_3[3,3],W_3[4,4] = 1,1,1,1,1
        W_3[5,5],W_3[6,6],W_3[7,7],W_3[8,8],W_3[9,9] = 1,1,1,1,1
        W_3[10,10],W_3[11,11],W_3[12,12],W_3[13,13],W_3[14,14] = 1,1,1,1,1
        W_3[15,15],W_3[16,16],W_3[17,17],W_3[18,18],W_3[19,19] = 1,1,1,1,1

        b_3 = torch.zeros_like(model[3].bias)

        W_5 = torch.zeros_like(model[5].weight)
        
        b_5 = torch.zeros_like(model[5].bias)

        model[1].weight = nn.Parameter(W_1)
        model[1].bias = nn.Parameter(b_1)
        model[3].weight = nn.Parameter(W_3)
        model[3].bias = nn.Parameter(b_3)
        model[5].weight = nn.Parameter(W_5)
        model[5].bias = nn.Parameter(b_5)


def fixSparseStructure_Iris(model):
    
    with torch.no_grad():

        W_1 = torch.zeros_like(model[1].weight)
        W_1[0,0] = 1
        W_1[1,1] = 1
        W_1[2,2] = 1
        W_1[3,3] = 1

        b_1 = torch.zeros_like(model[1].bias)

        W_3 = torch.zeros_like(model[3].weight)
        W_3[0,0] = 1
        W_3[1,1] = 1
        W_3[2,2] = 1
        W_3[3,3] = 1

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
