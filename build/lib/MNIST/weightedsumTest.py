import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from .OwnDescent import OwnDescent
from .helperFunctions import computeL1Norm
from .DataLoader import load_data_mnist
import time
import pickle



#*********Load Data ***************#

X_train,y_train, X_test,y_test = load_data_mnist()


def get_accuracy(net, X, y):
    
    correct = 0
    total = 0
    with torch.no_grad():
        #for inputs,labels in zip(X,y):
        outputs = net(X)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Define the neural network (number of parameters/weights 16330)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(784, 20)
        self.Linear2 = nn.Linear(20,20)
        self.Linear3 = nn.Linear(20,10)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        x = torch.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x
    

def weightedSum_methodMNIST(X_train,y_train, X_test,y_test):
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the weights for the objectives
    weights = np.linspace(0, 1, 44) # weights
    length = torch.tensor(len(weights))
    #print(len(weights))



    train_loss_values = []
    l1_norm_values = []
    test_loss_values = []

    for w in weights:

        model = Net()

        # Fix sparse neural network structure
        #fixSparseStructure1(model)

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)  
        # Train the network
        for epoch in tqdm.trange(190):
            # Forward pass
            outputs = model(X_train)
            loss_ = loss_fn(outputs, y_train)    

            # Compute the L1 norm of the weights and biases

            l1_norm = torch.abs(model.Linear1.weight).sum() + torch.abs(model.Linear1.bias).sum()
            l1_norm += torch.abs(model.Linear2.weight).sum() + torch.abs(model.Linear2.bias).sum()
            l1_norm += torch.abs(model.Linear3.weight).sum() + torch.abs(model.Linear3.bias).sum()
            
            
            # Weighted sum 
            loss = w * loss_ + (1-w) * l1_norm/length
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # model prediction
        y_pred_val = model(X_train)    
        train_loss = loss_fn(y_pred_val, y_train).item() 
        train_loss_values.append(train_loss)

        # Compute the L1 norm of the weights and biases
        l1_norm = torch.abs(model.Linear1.weight).sum() + torch.abs(model.Linear1.bias).sum()
        l1_norm += torch.abs(model.Linear2.weight).sum() + torch.abs(model.Linear2.bias).sum()
        l1_norm += torch.abs(model.Linear3.weight).sum() + torch.abs(model.Linear3.bias).sum()
        l1_norm = l1_norm/length
        l1_norm_values.append(l1_norm.item()) 


        w_1 = model.Linear1.weight.clone().detach()
        b_1 = model.Linear1.bias.clone().detach()
        w_3 = model.Linear2.weight.clone().detach()
        b_3 = model.Linear2.bias.clone().detach()
        w_5 = model.Linear3.weight.clone().detach()
        b_5 = model.Linear3.bias.clone().detach()

        Weights_all = [w_1, b_1, w_3, b_3, w_5, b_5]

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            # Compute loss
            y_pred = model(X_test)
                #_, predicted = torch.max(y_pred, 1)
            test_loss = loss_fn(y_pred, y_test).item()
            test_loss_values.append(test_loss)

    #print("train", train_loss_values)
    #l1_norm_values =[x / 20 for x in l1_norm_values]

    return   train_loss_values,l1_norm_values  



"""
# Plot the Pareto front
plt.rc('font', size=20, weight = "bold")  # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

val_loss_values,l1_norm_values = weightedSum_methodMNIST(X_train,y_train,X_valid, y_valid , X_test,y_test)
#plt.plot(val_loss_values,l1_norm_values,linestyle = '-',marker='o', color= "black",linewidth =7,markersize= 10,markerfacecolor = "red", markeredgecolor ="red",label='Pareto front')
plt.plot(val_loss_values,l1_norm_values,linestyle = '-',marker='o', color= "black",label='Pareto front')
plt.ylabel('L1 Norm') #fontsize = 15, fontweight='bold')
plt.xlabel('loss') #, fontsize = 15, fontweight='bold')
plt.title("Deterministic Pareto front Weighted Sum ", color="red", fontweight='bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.savefig("images/Deter_WeightedSum_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show()   

"""



def weightedSum_methodMNIST_stochastic(X_train,y_train, X_test,y_test):
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float('inf')

    # Define the weights for the objectives
    weights = np.linspace(0, 1, 44) # weights
    length = torch.tensor(len(weights))
    batch_num = 5600 #100 data points
    #print(len(weights))

    # ********** Accuracy**********
    train_acc_all = []
    test_acc_all = []
    

    model = Net()

    #***************************************************
    start = time.time()

    train_loss_values = []
    l1_norm_values = []
    test_loss_values = []

    for w in weights:

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)  

        # Train the network
        
        for epoch in tqdm.trange(200): #38
            train_loss = 0.0
            model.train()
            # Shuffle the training data
            permutation = torch.randperm(X_train.shape[0])


            # Split the training data into mini-batches
            for i in range(0, X_train.shape[0], batch_num):
                indices = permutation[i:i+batch_num]
                batch_X, batch_y = X_train[indices], y_train[indices]

                # Forward pass
                outputs = model(batch_X)
                loss_ = loss_fn(outputs, batch_y)    

                # Compute the L1 norm of the weights and biases

                l1_norm = torch.abs(model.Linear1.weight).sum() + torch.abs(model.Linear1.bias).sum()
                l1_norm += torch.abs(model.Linear2.weight).sum() + torch.abs(model.Linear2.bias).sum()
                l1_norm += torch.abs(model.Linear3.weight).sum() + torch.abs(model.Linear3.bias).sum()
                
                
                # Weighted sum 
                loss = w * loss_ + (1-w) * l1_norm/length
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss_.item()

     
        # model prediction
        y_pred = model(X_train) 
        train_loss = loss_fn(y_pred, y_train).item() 
        train_loss_values.append(train_loss)
        train_acc = get_accuracy(model, X_train, y_train)
        train_acc_all.append(train_acc)
        
        # Compute the L1 norm of the weights and biases
        l1_norm = torch.abs(model.Linear1.weight).sum() + torch.abs(model.Linear1.bias).sum()
        l1_norm += torch.abs(model.Linear2.weight).sum() + torch.abs(model.Linear2.bias).sum()
        l1_norm += torch.abs(model.Linear3.weight).sum() + torch.abs(model.Linear3.bias).sum()
        l1_norm = l1_norm/length
        l1_norm_values.append(l1_norm.item()) 

         # Print epoch statistics
        #print('Epoch %d | Train Loss: %.3f, Train Acc: %.3f | Val Loss: %.3f, Val Acc: %.3f' % (epoch+1, train_loss, train_acc, val_loss, val_acc))

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            # Compute test loss
            y_pred = model(X_test)
            test_loss = loss_fn(y_pred, y_test).item()
            test_loss_values.append(test_loss)
            test_acc= get_accuracy(model, X_test, y_test)
            test_acc_all.append(test_acc)
            #print("test_acc", test_acc)
    end = time.time()
    print("time.....................", end-start)
    
    return   train_loss_values,l1_norm_values, test_loss_values,train_acc_all,test_acc_all  


"""
# Plot the Pareto front
train_loss_values,l1_norm_values,val_loss_values,test_loss_values,train_acc_all,test_acc_all = weightedSum_methodMNIST_stochastic(X_train,y_train,X_valid, y_valid , X_test,y_test)
plt.plot(train_loss_values,l1_norm_values , '--o', color= "black", label='WS_train')
plt.plot(val_loss_values,l1_norm_values, '--o', color= "red", label='WS_valid')
plt.plot(test_loss_values,l1_norm_values, '--o', color= "yellow", label='WS_test')
plt.title("Stochastic Pareto front Weighted Sum ")
plt.ylabel('L1 Norm')
plt.xlabel('loss')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
   
plt.savefig("images/Stoch_WeightedSum_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None) 
plt.show()


plt.plot(train_acc_all, l1_norm_values,"--o", color= "green", label = "WS_train_acc" )
plt.plot(test_acc_all, l1_norm_values,"--o", color= "brown", label = "WS_test_acc" )
plt.ylabel("l1 norm")
plt.xlabel("accuracy")
plt.title ('Accuray plot for WeightedSum Pareto front')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.savefig("images/Accuracy_WeightedSum_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show() 
"""