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
import pickle
import tqdm
import torch.optim as optim
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
np.random.seed(164) #148 268


#**********SETUP AND INITIALIZATIONS*****************'

# number of runs for the predictor-corrector algorithm (CM)
n_continuation= 1

# number of training epochs for first run to obtain initial point on the front
n_corr_first= 2000 


#******L1-norm objective**********
# number of points that hopefully belong to the pareto front 
n_pareto_prox= 10
# number of iterations for predictor step (shrinkage)
n_predictor=  2
# Set number of training epochs for corrector step (Algorithm 1) for L1 objective function
n_corr= 25


#******loss objective**********
# number of points that hopefully belong to the pareto front (loss objective)
n_pareto_grad= 10 
# number of iterations for predictor step (gradient)
n_predictor_loss= 7
#number of iterations for corrector step (Algorithm 1)
n_corrector_loss= 25

##################################################################################



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

X_train,y_train, X_test,y_test = load_data_cifar10()

##################################################################################


#**************MODEL***************#
model =nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 32, kernel_size=4),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(32*6*6, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.LogSoftmax(dim=1)
)

# Save the initial parameters to a file
torch.save(model.state_dict(), 'initial_cnn_model.pth')
##################################################################################

class OwnDescent(Optimizer):
    """Class for the multiproximal gradient optimizers.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        lr: learning rate
        sr: shrinkage rate
        alpha: momentum rate
        eps: stopping criterion
    """

    def __init__(self, params, lr, sr, alpha):
        if lr is None or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, sr=sr)

        super(OwnDescent, self).__init__(params, defaults)

        self.alpha = alpha
        # parameters for nesterov acceleration
        self.k = 0
        # our model has only on param_group so this is okay
        self.last_p = self.param_groups[0]['params'].copy()


    def __setstate__(self, state):
        super(OwnDescent, self).__setstate__(state)

    @torch.no_grad()
    def acceleration_step(self):
        """Performs an accelerated gradient optimization step for Iris data.
        """

        # store l1 norm of current parmeters
        p_ = [torch.clone(p) for p in self.param_groups[0]['params']]
        l1norm_p_current = sum([torch.sum(torch.abs(p__)) for p__ in p_])

        p_old = []

        for group in self.param_groups:

            lr = group['lr']

            for i, p in enumerate(group['params']):

                #save last iteration in case of restart
                p_old.append(torch.clone(p))

                #acceleration
                acc_step = torch.add(p, self.last_p[i], alpha=-1)
                acc_alpha = (self.k -1)/(self.alpha+self.k)
                acc_step = torch.mul(acc_step, acc_alpha)
                p.add_(acc_step, alpha=1)

                #update for acceleration
                #use torch.clone() to create a copy
                self.last_p[i] = torch.clone(p)

      

        l1norm_p_acc = sum([torch.sum(torch.abs(p__)) for p__ in self.param_groups[0]['params']])

        # restart acceleration scheme if l1 norm is increased by acceleration step
        if l1norm_p_acc > l1norm_p_current:
            # undo acceleration step

            for group in self.param_groups:

                for i, p in enumerate(group['params']):

                    p.copy_(p_old[i])

                    #use torch.clone() to create a copy
                    self.last_p[i] = torch.clone(p)

            self.k = 0

        else:

            # update acceleration
            self.k = self.k + 1

    @torch.no_grad()
    def shrinkage(self):

        """Performs a shrinkage step from equation 4 in our paper.
        """

        for group in self.param_groups:
            
            sr = group['sr']

            for p in group['params']:

                # shrinkage operator
                c = torch.mul(torch.sign(p), torch.max(torch.add(torch.abs(p), torch.ones_like(p), alpha = -sr), torch.zeros_like(p)))
                p.add_(p, alpha = -1)
                p.add_(c, alpha=1)


    @torch.no_grad()
    def MOOproximalgradientstep(self):

        """Performs a multiobjective proximal gradient step using Algorithm 1 in our paper.
        """

        for group in self.param_groups:
            
            lr = group['lr']

            x = [p.clone().detach() for p in group['params']]
            d = [p.grad.clone().detach() for p in group['params']]


            x, structure = self.stackTensor(x)
            d = self.stackTensor(d)[0]

            Y = self.MOOproximalgradientUpdate(x, d, lr)
            Y = self.convertStackedTensorToStructuredTensor(Y, structure)

            for p, y in zip(group['params'], Y):

                p.add_(p, alpha = -1)
                p.add_(y, alpha=1)
      


    @torch.no_grad()
    def MOOproximalgradientUpdate(self, x, d, h):
        """Performs an update.
        """
        
        # step should be a torch tensor of dimension 1 times 
        
        y_ = lambda alpha : torch.mul(torch.sign(x.add(d, alpha = -h*alpha)), torch.max(torch.add(torch.abs(x.add(d, alpha = -h*alpha)), torch.ones_like(x), alpha = -h*(1-alpha)), torch.zeros_like(x)))

        omega_1 = lambda alpha : torch.dot(d, y_(alpha) - x)

        l1_x = torch.norm(x, p = 1)

        omega_2 = lambda alpha : torch.norm(y_(alpha), p = 1) - l1_x 


        alpha = .5
 
        for j in range(10):

            if omega_1(alpha) > omega_2(alpha):

                alpha = alpha + (.5)**(j+2)

            else:

                alpha = alpha - (.5)**(j+2)

        y = y_(alpha)

        if alpha > .25 and alpha < .75:
            print('yo')

        return y

    def stackTensor(self, tensor):

        """save size of each tensor in a list for reconstruction purposes
        """

        structure = [tensor_.size() for tensor_ in tensor]

        stacked_tensor = torch.cat([tensor_.reshape(-1) for tensor_ in tensor])

        return stacked_tensor, structure


    def convertStackedTensorToStructuredTensor(self, tensor_direction, structure):
        # Create an empty list to store the structured tensors
        tensors = []
        index = 0  # Initialize the index for slicing the direction tensor

        # Iterate through the shapes in the structure list
        for s in structure:
            # Calculate the total size of the tensor based on its shape
            size = torch.prod(torch.tensor(s)).item()

            # Slice the direction tensor to get the data for the current tensor
            tensor_data = tensor_direction[index:index + size]

            # Reshape the sliced data into the desired shape
            structured_tensor = tensor_data.view(*s)

            # Append the structured tensor to the list
            tensors.append(structured_tensor)

            # Update the index for the next slice
            index += size

        # Check if there are remaining elements in the direction tensor
        if index < len(tensor_direction):
            print('Conversion of tensor direction from list to structured tensor failed. There are remaining elements in the list!')

        return tensors

##################################################################################

#**************DEFININIG ADDITIONAL FUNCTIONS******************#
def computeL1Norm(model):

    with torch.no_grad():
        L1 = 0

        L1 = L1 + torch.sum(torch.abs(model[0].weight))
        L1 = L1 + torch.sum(torch.abs(model[0].bias))
        L1 = L1 + torch.sum(torch.abs(model[3].weight))
        L1 = L1 + torch.sum(torch.abs(model[3].bias))
        L1 = L1 + torch.sum(torch.abs(model[7].weight))
        L1 = L1 + torch.sum(torch.abs(model[7].bias))
        L1 = L1 + torch.sum(torch.abs(model[9].weight))
        L1 = L1 + torch.sum(torch.abs(model[9].bias))
    

    return L1

def get_accuracy(model, X, y, batch_size=64):
    """Performs a measure of the correct prediction made

       Args: 
          model: neural network model used
          X, y: Could be either the training dataset or testing dataset
          batch_size: Size of each batch, default 64
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        # Split the data into batches
        for i in range(0, len(X), batch_size):
            inputs = X[i:i+batch_size].to(device)
            labels = y[i:i+batch_size].to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    #model.train()  # Set the model back to training mode
    return accuracy

def get_accuracy1(model, X, y):
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
##################################################################################


####*********LOADING MODEL AND DATA TO GPU***********************#
model = model.to(device)
X_train,y_train, X_test,y_test = X_train.to(device),y_train.to(device), X_test.to(device),y_test.float().to(device)

##################################################################################





#batch_num = 4800 #10 data point in each batch ie. batch size = 10
#batch_num = 3000 #16 data point in each batch ie. batch size = 16
#batch_num = 1500 #32 data point in each batch ie. batch size = 32
batch_num = 750 #64 data point in each batch ie. batch size = 64
#batch_num = 375 #128 data point in each batch ie. batch size = 128
scaler = GradScaler()
train_acc_all = []
test_acc_all = []

for k_ in range(n_continuation):

    # Create Optimizer
    params = model.parameters()
    learning_rate =0.001 #0.001 good
    shrinkage_rate = 5e-3
    optimizer = OwnDescent(params, lr=learning_rate, sr=shrinkage_rate, alpha = 2)
    optimizer_adam = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-8)
    #optimizer_adam  = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

    loss = nn.CrossEntropyLoss()



    L1Norm_start = np.zeros((n_corr_first+1,))
    train_loss_start = np.zeros((n_corr_first+1,))
    


    # first run to get first point on the pareto front
    for epoch in tqdm.trange(n_corr_first):
     
        model.train()
        # Shuffle the training data
        permutation = torch.randperm(X_train.shape[0])
        # Split the training data into mini-batches
        for i in range(0, X_train.shape[0], batch_num):
            indices = permutation[i:i+batch_num]
            batch_X, batch_y = X_train[indices], y_train[indices]
            optimizer.acceleration_step()
            #with autocast():
            # model prediction
            y_pred = model(batch_X).float()
            # compute loss
            Loss = loss(y_pred, batch_y.long())
            weight_length_ = computeL1Norm(model)/ (Loss.item() *10)
            # store values for potential pareto point
            L1Norm_start[epoch] = computeL1Norm(model)/weight_length_
            train_loss_start[epoch] = Loss.item()

            # compute gradients
            optimizer.zero_grad()
            Loss.backward()

            # preform moo proximal gradient step
            optimizer.MOOproximalgradientstep()

            val_accuracy = get_accuracy(model, batch_X, batch_y)

        #print(f"Epoch {epoch+1}/{n_corr_first}, Loss: {Loss.item()}, train Accuracy: {val_accuracy}")

    #model prediction
    with autocast():
        y_pred = model(X_train).float()
        # compute loss
        Loss = loss(y_pred, y_train.long())
        weight_length = computeL1Norm(model)/ (Loss.item() *10) # For normalizing the weights
        # store values for potential pareto point
        L1Norm_start[n_corr_first] = computeL1Norm(model)/weight_length
        train_loss_start[n_corr_first] = Loss.item()
        train_acc = get_accuracy(model, X_train, y_train)
        train_acc_all.append(train_acc)
    with open("length_number.txt", "w") as file:
        file.write(str(weight_length.item()))

    print(f"Loss: {train_loss_start[n_corr_first]},  Train Accuracy first_run : {train_acc}")

    model.eval()
    test_loss_start = np.zeros((n_corr_first+1,))
    with torch.no_grad():
        # Split the data into batches
        for i in range(0, len(X_test), batch_num):
            batch_Xt, batch_yt = X_test[i:i+batch_num],y_test[i:i+batch_num]
            # Compute testing accuracy
            y_pred = model(batch_Xt).float()
            test_loss_start[n_corr_first] = loss(y_pred, batch_yt.long()).item()
            test_acc = get_accuracy1(model, batch_Xt, batch_yt)
        test_acc_all.append(test_acc)
        print(f"Test Loss: {test_loss_start[n_corr_first] }, Test Accuracy first_run : {test_acc}")

    # Store model weights for second predictor corrector approach from potential pareto critical point

    #*******Saving the weights to memory**********
    W_0_start = model[0].weight.clone().detach()
    b_0_start = model[0].bias.clone().detach()
    W_3_start = model[3].weight.clone().detach()
    b_3_start = model[3].bias.clone().detach()
    W_7_start = model[7].weight.clone().detach()
    b_7_start = model[7].bias.clone().detach()
    W_9_start = model[9].weight.clone().detach()
    b_9_start = model[9].bias.clone().detach()


    Weights_start = [ W_0_start,b_0_start,W_3_start,b_3_start,W_7_start,b_7_start,W_9_start,b_9_start]#,W_11_start,b_11_start]

    L1Norm_pred_grad_all = []
    train_loss_pred_grad_all = []
    

    L1Norm_corr_grad_all = []
    train_loss_corr_grad_all = []
    test_loss_corr_grad_all = []

    # continuation method
    # outer loop for predictor
    # inner loop for corrector
    # predictor = gradient step for loss
    for pareto in tqdm.trange(n_pareto_grad):

        L1Norm_pred = np.zeros((n_predictor_loss+1,))
        train_loss_pred = np.zeros((n_predictor_loss+1,))


        model.train()
        # perform a number of gradient steps for prediction
        for pred in range(n_predictor_loss):
            #model.train()
            # Shuffle the training data
            permutation = torch.randperm(X_train.shape[0])

            # Split the training data into mini-batches
            for i in range(0, X_train.shape[0], batch_num):
                indices = permutation[i:i+batch_num]
                batch_X, batch_y = X_train[indices], y_train[indices]

                # model prediction
                y_pred = model(batch_X).float()
                # compute loss
                Loss = loss(y_pred, batch_y.long())
                # store values for potential pareto point
                L1Norm_pred[pred] = computeL1Norm(model)/weight_length
                train_loss_pred[pred] = Loss.item()

                # compute gradient
                optimizer_adam.zero_grad()
                Loss.backward()
                # perform gradient step
                optimizer_adam.step()

        # model prediction
        #model.eval()
        with autocast():
            y_pred = model(X_train).float()
            # compute loss
            Loss = loss(y_pred, y_train.long())
            # store values for potential pareto point
            L1Norm_pred[n_predictor_loss] = computeL1Norm(model)/weight_length
            train_loss_pred[n_predictor_loss] = Loss.item()
            train_acc = get_accuracy(model, X_train, y_train)
            print(f"Loss: {Loss.item()}, Train Accuracy predictor Adam : {train_acc}")

        
        L1Norm_pred_grad_all.append(L1Norm_pred.copy())
        train_loss_pred_grad_all.append(train_loss_pred.copy())

        
        #*** Corrector step for Loss*************#
        L1Norm_corr = np.zeros((n_corrector_loss+1,))
        train_loss_corr = np.zeros((n_corrector_loss+1,))
        test_loss_corr = np.zeros((n_corrector_loss+1,))



        for corr in tqdm.trange(n_corrector_loss):
            model.train()
            # Shuffle the training data
            permutation = torch.randperm(X_train.shape[0])

            # Split the training data into mini-batches
            for i in range(0, X_train.shape[0], batch_num):
                indices = permutation[i:i+batch_num]
                batch_X, batch_y = X_train[indices], y_train[indices]

                optimizer.acceleration_step()

                # model predction
                y_pred = model(batch_X).float()
                # compute loss
                Loss = loss(y_pred, batch_y.long())

                # store values for potential pareto point
                L1Norm_corr[corr] = computeL1Norm(model)/weight_length
                train_loss_corr[corr] = Loss.item()

                # compute gradient
                optimizer.zero_grad()
                Loss.backward()

                # preform moo proximal gradient step
                optimizer.MOOproximalgradientstep()

        #model predction
        #model.eval()
        with autocast():
            y_pred = model(X_train).float()
            # compute loss
            Loss = loss(y_pred, y_train.long())
            # store values for potential pareto point
            L1Norm_corr[n_corrector_loss] = computeL1Norm(model)/weight_length
            train_loss_corr[n_corrector_loss] = Loss.item()
            train_acc = get_accuracy(model, X_train, y_train)
            train_acc_all.append(train_acc)
            print(f"Loss: {Loss.item()}, Train Accuracy Corrector Lossfn : {train_acc}")

        model.eval()
        with torch.no_grad():
            # Split the data into batches
            for i in range(0, len(X_test), batch_num):
                batch_Xt, batch_yt = X_test[i:i+batch_num],y_test[i:i+batch_num]
                y_pred = model(batch_Xt).float()
                test_loss_corr[n_corrector_loss] = loss(y_pred, batch_yt.long()).item()
                test_acc = get_accuracy1(model, batch_Xt, batch_yt)
        test_acc_all.append(test_acc)
        # print('Test Accuracy: %.3f' % test_acc)
        print(f"Test Loss: {test_loss_corr[n_corrector_loss]}, Test Accuracy corrector Lossfn : {test_acc}")


        L1Norm_corr_grad_all.append(L1Norm_corr.copy())
        train_loss_corr_grad_all.append(train_loss_corr.copy())
        test_loss_corr_grad_all.append(test_loss_corr.copy())

    test_loss_corr_grad_all = test_loss_corr_grad_all[::-1]
    test_acc_all = test_acc_all[::-1]
    train_acc_all = train_acc_all[::-1]
    L1Norm_corr_grad_all = L1Norm_corr_grad_all[::-1]
    train_loss_corr_grad_all =train_loss_corr_grad_all[::-1]    

    W_0_pred_grad = model[0].weight.clone().detach()
    b_0_pred_grad = model[0].bias.clone().detach()
    W_3_pred_grad = model[3].weight.clone().detach()
    b_3_pred_grad = model[3].bias.clone().detach()
    W_7_pred_grad= model[7].weight.clone().detach()
    b_7_pred_grad = model[7].bias.clone().detach()
    W_9_pred_grad = model[9].weight.clone().detach()
    b_9_pred_grad = model[9].bias.clone().detach()


    Weights_pred_grad = [ W_0_pred_grad,b_0_pred_grad,W_3_pred_grad,b_3_pred_grad,W_7_pred_grad,b_7_pred_grad,W_9_pred_grad,b_9_pred_grad]#,W_11_pred_grad,b_11_pred_grad]


    # set weights to weights that where computed from start iteration

    for group in optimizer.param_groups:

        with torch.no_grad():

            group['params'][0].copy_(nn.Parameter(W_0_start))
            group['params'][1].copy_(nn.Parameter(b_0_start))
            group['params'][2].copy_(nn.Parameter(W_3_start))
            group['params'][3].copy_(nn.Parameter(b_3_start))
            group['params'][4].copy_(nn.Parameter(W_7_start))
            group['params'][5].copy_(nn.Parameter(b_7_start))
            group['params'][6].copy_(nn.Parameter(W_9_start))
            group['params'][7].copy_(nn.Parameter(b_9_start))



    L1Norm_pred_prox_all = []
    train_loss_pred_prox_all = []


    L1Norm_corr_prox_all = []
    train_loss_corr_prox_all = []
    test_loss_corr_prox_all = []

    # continuation method
    # outer loop for predictor
    # inner loop for corrector
    # predictor = shrinkage step for L1-Norm
    for pareto in tqdm.trange(n_pareto_prox):

        L1Norm_pred = np.zeros((n_predictor+1,))
        train_loss_pred = np.zeros((n_predictor+1,))
        

        # perform a number of gradient steps for prediction
        model.train()
        for pred in range(n_predictor):

            # model prediction
            y_pred = model(X_train).float()
            # compute loss
            Loss = loss(y_pred, y_train.long())
            # store values for potential pareto point
            L1Norm_pred[pred] = computeL1Norm(model)/weight_length
            train_loss_pred[pred] = Loss.item()

            # compute gradient
            optimizer.zero_grad()
            Loss.backward()
            # perform gradient step
            optimizer.shrinkage()

        # model prediction
        with autocast():
            y_pred = model(X_train).float()
            # compute loss
            Loss = loss(y_pred, y_train.long())
            # store values for potential pareto point
            L1Norm_pred[n_predictor] = computeL1Norm(model)/weight_length
            train_loss_pred[n_predictor] = Loss.item()


        L1Norm_pred_prox_all.append(L1Norm_pred.copy())
        train_loss_pred_prox_all.append(train_loss_pred.copy())


        L1Norm_corr = np.zeros((n_corr+1,))
        train_loss_corr = np.zeros((n_corr+1,))
       
        model.train()
        # inner loop for correction
        for corr in tqdm.trange(n_corr):
           
            # Shuffle the training data
            permutation = torch.randperm(X_train.shape[0])

            # Split the training data into mini-batches
            for i in range(0, X_train.shape[0], batch_num):
                indices = permutation[i:i+batch_num]
                batch_X, batch_y = X_train[indices], y_train[indices]

                optimizer.acceleration_step()

                # model predction
                y_pred = model(batch_X).float()
                # compute loss
                Loss = loss(y_pred, batch_y.long())

                # store values for potential pareto point
                L1Norm_corr[corr] = computeL1Norm(model)/weight_length
                train_loss_corr[corr] = Loss.item()

                # compute gradient
                optimizer.zero_grad()
                Loss.backward()

                # preform moo proximal gradient step
                optimizer.MOOproximalgradientstep()


        # model predction
        #model.eval()
        with autocast():
            y_pred = model(X_train).float()
            # compute loss
            Loss = loss(y_pred, y_train.long())
            # store values for potential pareto point
            L1Norm_corr[n_corr] = computeL1Norm(model)/weight_length
            train_loss_corr[n_corr] = Loss.item()
            train_acc = get_accuracy(model, X_train, y_train)
            train_acc_all.append(train_acc)


        ##### Testing *************
        test_loss_corr = np.zeros((n_corr+1,))
        model.eval()
        with torch.no_grad():
            # Split the data into batches
            for i in range(0, len(X_test), batch_num):
                batch_Xt, batch_yt = X_test[i:i+batch_num],y_test[i:i+batch_num]
                y_pred = model(batch_Xt).float()
                test_loss_corr[n_corr] = loss(y_pred, batch_yt.long()).item()
                test_acc = get_accuracy(model, batch_Xt, batch_yt)
        test_acc_all.append(test_acc)


        L1Norm_corr_prox_all.append(L1Norm_corr.copy())
        train_loss_corr_prox_all.append(train_loss_corr.copy())
        test_loss_corr_prox_all.append(test_loss_corr.copy())

    W_0_pred_prox = model[0].weight.clone().detach()
    b_0_pred_prox = model[0].bias.clone().detach()
    W_3_pred_prox = model[3].weight.clone().detach()
    b_3_pred_prox = model[3].bias.clone().detach()
    W_7_pred_prox = model[7].weight.clone().detach()
    b_7_pred_prox = model[7].bias.clone().detach()
    W_9_pred_prox = model[9].weight.clone().detach()
    b_9_pred_prox = model[9].bias.clone().detach()


    Weights_pred_prox = [ W_0_pred_prox,b_0_pred_prox,W_3_pred_prox,b_3_pred_prox,W_7_pred_prox,b_7_pred_prox,W_9_pred_prox,b_9_pred_prox]#, W_11_pred_prox,b_11_pred_prox]


    Weights_all = [Weights_start, Weights_pred_grad, Weights_pred_prox]

    L1Norm_all = [L1Norm_start, L1Norm_pred_grad_all, L1Norm_corr_grad_all, L1Norm_pred_prox_all, L1Norm_corr_prox_all]
    train_loss_all = [train_loss_start, train_loss_pred_grad_all, train_loss_corr_grad_all, train_loss_pred_prox_all, train_loss_corr_prox_all]
    test_loss_all = [test_loss_start, test_loss_corr_grad_all, test_loss_corr_prox_all]
    accuracy = [train_acc_all,test_acc_all]


    # Parent Directory path
    parent_dir = os.path.join(os.getcwd(), 'Results')


    directory_names = [dir for dir in os.walk(parent_dir)]
    directory_names = [dir[0] for dir in directory_names[1::]]
    L = len(directory_names) # L number of result files

    # Directory
    directory = "Results_cm_sto" 

    # Path
    path = os.path.join(parent_dir, directory)

    # create directory in path
    if not os.path.exists(path):
        os.mkdir(path)

    L1Norm_path = os.path.join(path, "L1NormAll")
    TrainLoss_path = os.path.join(path, "TrainLoss")
    TestLoss_path = os.path.join(path, "TestLoss")
    accuracy_path = os.path.join(path, "accuracy")

    with open(L1Norm_path, "wb") as fp:   #Pickling
        pickle.dump(L1Norm_all, fp)
    with open(TrainLoss_path, "wb") as fp:   #Pickling
        pickle.dump(train_loss_all, fp)
    with open(TestLoss_path, "wb") as fp:   #Pickling
        pickle.dump(test_loss_all, fp)
    with open(accuracy_path, "wb") as fp:   #Pickling
        pickle.dump(accuracy, fp)

    Text_file_path = os.path.join(path, "Info.txt")

    with open(Text_file_path, 'w') as f:

        f.write('# number of runs of continuation method runs\n')
        f.write(f'n_continuation = {n_continuation}\n')

        f.write('# number of points that hopefully belong to the pareto front for grad continuation\n')
        f.write(f'n_pareto = {n_pareto_grad}\n')

        f.write('# number of points that hopefully belong to the pareto front for prox continuation\n')
        f.write(f'n_pareto = {n_pareto_prox}\n')

        f.write('# numper of iterations for prediction for loss\n')
        f.write(f'n_predictor = {n_predictor_loss}\n')

        f.write('# number of training epochs for corrector for loss\n')
        f.write(f'n_corr = {n_corrector_loss}\n')

        f.write('# numper of iterations for prediction for l1-norm\n')
        f.write(f'n_predictor = {n_predictor}\n')

        f.write('# number of training epochs for corrector for l1-norm\n')
        f.write(f'n_corr = {n_corr}\n')

        f.write('# number of training epochs for first run\n')
        f.write(f'n_corr_first = {n_corr_first}\n')

        f.write('\n')
        f.write(f'Training loss after start iteration = {train_loss_start[-1]}\n')
        f.write(f'Test loss after start iteration = {test_loss_start[-1]}\n')
        f.write(f'L1 norm after start iteration = {L1Norm_start[-1]}\n')

        f.write('\n')
        f.write(f'Training loss after grad continuation = {train_loss_corr_grad_all[-1][-1]}\n')
        f.write(f'Test loss after grad continuation = {test_loss_corr_grad_all[-1][-1]}\n')
        f.write(f'L1 norm after grad continuation = {L1Norm_corr_grad_all[-1][-1]}\n')

        f.write('\n')
        f.write(f'Training loss after prox continuation = {train_loss_corr_prox_all[-1][-1]}\n')
        f.write(f'Test loss after prox continuation = {test_loss_corr_prox_all[-1][-1]}\n')
        f.write(f'L1 norm after prox continuation = {L1Norm_corr_prox_all[-1][-1]}\n')
        f.write('\n')

print("finish")
