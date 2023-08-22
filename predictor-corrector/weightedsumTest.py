import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from helperFunctions import  get_accuracy
from DataLoader import load_data_mnist
import time
import pickle



#*********Load Data ***************#

X_train,y_train, X_test,y_test = load_data_mnist()

# Define the neural network architechture (number of parameters/weights 16330)
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
    
def weightedSum_methodMNIST_stochastic(X_train,y_train, X_test,y_test):
    
    """Performs a stochastic training and testing on the mnist dataset using the weighted sum algorithm.
        Args:
            X_train,y_train, X_test,y_test: The splitted training and testing sets of 80-20 ratio.
    """
     
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()
   
    # Define the weights for the objectives
    weights = np.linspace(0, 1, 44) # 44 equidistantly distributed weights
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
        #print('Epoch %d | Train Loss: %.3f, Train Acc: %.3f (epoch+1, train_loss, train_acc))

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
        f.write(f'Total computation time for deterministic Train/Test = {end-start}\n')
        f.write('\n')

    print("finish")
  
    return   train_loss_values,l1_norm_values, test_loss_values,train_acc_all,test_acc_all  


#train_loss_values,l1_norm_values,test_loss_values,train_acc_all,test_acc_all = weightedSum_methodMNIST_stochastic(X_train,y_train,X_test,y_test)