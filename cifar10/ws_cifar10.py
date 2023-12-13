import torch
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import tqdm
import torch.optim as optim
import time
import pickle
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn
np.random.seed(164) # 148


#**************Checking your connection to GPU**************#

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

cudnn.benchmark = True

if use_cuda == False:

    print("WARNING: CPU will be used for training.")

##################################################################################


#**************Data Loading for CIFAR10 Data set********************#
(train_X, train_y), (test_X, test_y) = tf.keras.datasets.cifar10.load_data()

def load_data_cifar10():
    """Performs the loading and splitting of  CIFAR10 dataset into train and test sets.

       Returns:
            X_train,y_train, X_test,y_test: The splitted training and testing sets of 80-20 ratio. 
    """

    # Concatenate training and testing
    X = np.concatenate([train_X, test_X], axis=0).reshape((60000, 32*32*3))
    # print(X.shape)
    y = np.append(train_y, test_y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

     #*********** Split the data set into training and testing**********#
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.2,train_size=0.8)
    # Converting to torch
    X_train = Variable(torch.from_numpy(X_train)).reshape(48000, 3, 32, 32).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test  = Variable(torch.from_numpy(X_test)).reshape(12000, 3, 32, 32).float()
    y_test  = Variable(torch.from_numpy(y_test)).long()

    return X_train,y_train,X_test,y_test


#*********Load Data ***************#

X_train,y_train, X_test,y_test = load_data_cifar10()


# Define the neural network architechture (number of parameters/weights 4742546)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)  # Adjusted kernel size and added padding
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*6*6, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x) 
        return x
    
##################################################################################

#**************DEFININIG ADDITIONAL FUNCTIONS******************#

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



def weightedSum_methodMNIST_stochastic(X_train,y_train, X_test,y_test):
    
    """Performs a stochastic training and testing on the mnist dataset using the weighted sum algorithm.
        Args:
            X_train,y_train, X_test,y_test: The splitted training and testing sets of 80-20 ratio.
    """
    
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()
   
    # Define the weights for the objectives
    
    weights = np.linspace(0, 1, 20) # 20 equidistantly distributed weights
    with open("length_number.txt", "r") as file:
        length_ = float(file.read())
    length = torch.tensor(length_)
    #print("length: ",length)

    batch_num = 750 #64 data point in each batch ie. batch size = 64

    # ********** Accuracy**********
    train_acc_all = []
    test_acc_all = []

    
    #***************************************************
    start = time.time()

    train_loss_values = []
    l1_norm_values = []
    test_loss_values = []


    for i, w in enumerate(weights):
        #print(w)

        model = Net()
        # Load the initial weight parameters from CM
        initial_state_dict = torch.load('initial_cnn_model.pth')
        # Load the state dictionary with the consistent names
        state_dict_b = model.state_dict()
        # Map keys from Sequential model to Custom Class model
        mapped_state_dict = {
            'conv1.weight': initial_state_dict['0.weight'],
            'conv1.bias': initial_state_dict['0.bias'],
            'conv2.weight': initial_state_dict['3.weight'],
            'conv2.bias': initial_state_dict['3.bias'],
            'fc1.weight': initial_state_dict['7.weight'],
            'fc1.bias': initial_state_dict['7.bias'],
            'fc2.weight': initial_state_dict['9.weight'],
            'fc2.bias': initial_state_dict['9.bias'],
        }
        # Load the mapped state dictionary into the custom class model
        model.load_state_dict(mapped_state_dict)

        ####*********LOADING MODEL AND DATA TO GPU***********************#
        model = model.to(device) 
        X_train,y_train, X_test,y_test = X_train.to(device),y_train.to(device), X_test.to(device),y_test.float().to(device)

        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # Train the network
        for epoch in tqdm.trange(118): 
            train_loss = 0.0
            model.train()
            # Shuffle the training data
            permutation = torch.randperm(X_train.shape[0])


            # Split the training data into mini-batches
            for i in range(0, X_train.shape[0], batch_num):
                indices = permutation[i:i+batch_num]
                batch_X, batch_y = X_train[indices], y_train[indices]

                # Forward pass
                outputs = model(batch_X).float()
                loss_ = loss_fn(outputs, batch_y.long())    

                # Compute the L1 norm of the weights and biases

                l1_norm = torch.abs(model.conv1.weight).sum() + torch.abs(model.conv1.bias).sum()
                l1_norm += torch.abs(model.conv2.weight).sum() + torch.abs(model.conv2.bias).sum()
                l1_norm += torch.abs(model.fc1.weight).sum() + torch.abs(model.fc1.bias).sum()
                l1_norm += torch.abs(model.fc2.weight).sum() + torch.abs(model.fc2.bias).sum()

                # Weighted sum 
                loss = w * loss_ + (1-w) * (l1_norm/length)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss_.item()
            
        # model prediction
        with autocast():
            y_pred = model(X_train).float()
            train_loss = loss_fn(y_pred, y_train.long()).item() 
            train_loss_values.append(train_loss)
            train_acc = get_accuracy(model, X_train, y_train)
            train_acc_all.append(train_acc)

            # Compute the L1 norm of the weights and biases
            l1_norm = torch.abs(model.conv1.weight).sum() + torch.abs(model.conv1.bias).sum()
            l1_norm += torch.abs(model.conv2.weight).sum() + torch.abs(model.conv2.bias).sum()
            l1_norm += torch.abs(model.fc1.weight).sum() + torch.abs(model.fc1.bias).sum()
            l1_norm += torch.abs(model.fc2.weight).sum() + torch.abs(model.fc2.bias).sum()
            l1_norm = l1_norm/length
            l1_norm_values.append(l1_norm.item()) 

        # Print epoch statistics
        #print('Epoch %d | Train Loss: %.3f, Train Acc: %.3f (epoch+1, train_loss, train_acc))

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            # Compute test loss
            y_pred = model(X_test).float()
            test_loss = loss_fn(y_pred, y_test.long()).item()
            test_loss_values.append(test_loss)
            test_acc= get_accuracy(model, X_test, y_test)
            test_acc_all.append(test_acc)

        print("train_acc", train_acc)
        print("test_acc", test_acc)
        print("l1_norm", l1_norm) 

        
            
    end = time.time()
    #print("time.....................", end-start)

    #*********store values**********#
    # Parent Directory path
    parent_dir = os.path.join(os.getcwd(), 'Results')

    directory_names = [dir for dir in os.walk(parent_dir)]
    directory_names = [dir[0] for dir in directory_names[1::]]
    L = len(directory_names) # L number of result files

    # Directory
    directory = "Results_ws_sto"
    
    # Path
    path = os.path.join(parent_dir, directory)

    # create directory in path
    if not os.path.exists(path):
        os.mkdir(path)
        train_loss_values,l1_norm_values, test_loss_values,train_acc_all,test_acc_all  

    TrainLoss_path = os.path.join(path, "TrainLoss")
    L1Norm_path = os.path.join(path, "L1NormAll")
    TestLoss_path = os.path.join(path, "TestLoss")
    TrainAccuracy_path = os.path.join(path, "TrainAccuracy")
    TestAccuracy_path = os.path.join(path, "TestAccuracy")

    with open(TrainLoss_path, "wb") as fp:   #Pickling
        pickle.dump(train_loss_values, fp)
    with open(L1Norm_path, "wb") as fp:   #Pickling
        pickle.dump(l1_norm_values, fp)
    with open(TestLoss_path, "wb") as fp:   #Pickling
        pickle.dump(test_loss_values, fp)

    with open(TrainAccuracy_path, "wb") as fp:   #Pickling
        pickle.dump(train_acc_all, fp)
    with open(TestAccuracy_path, "wb") as fp:   #Pickling
        pickle.dump(test_acc_all , fp)


    Text_file_path = os.path.join(path, "Info.txt")

    with open(Text_file_path, 'w') as f:
        f.write('\n')
        f.write(f'Training loss for weighted sum = {train_loss_values[-1]}\n')
        f.write(f'Test loss for weighted sum = {test_loss_values[-1]}\n')
        f.write(f'Training accuracy for weighted sum = {train_acc_all[-1]}\n')
        f.write(f'Testing accuracy for weighted sum= {test_acc_all[-1]}\n')
        f.write('\n')
        f.write(f'Total computation time for stochastic Train/Test = {end-start}\n')
        f.write('\n')

    print("finish")
  
    return   train_loss_values,l1_norm_values, test_loss_values,train_acc_all,test_acc_all  


train_loss_values,l1_norm_values,test_loss_values,train_acc_all,test_acc_all = weightedSum_methodMNIST_stochastic(X_train,y_train,X_test,y_test)

