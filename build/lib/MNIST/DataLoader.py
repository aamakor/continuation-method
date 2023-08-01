import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from fast_ml.model_development import train_valid_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import yaml
from importlib import resources
import io


# Load MNIST Data into np arrays
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Load Iris Data
iris = load_iris()


# Data Visualization
def visualize_training_data():
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

    for i in range(1, cols*rows +1):
        # takes random sample (using random sample index)
        sample_idx = torch.randint(len(train_X), size=(1,)).item()

        # Training data accessed by indexing returns img, label-tuple
        img, label = train_X[sample_idx], train_y[sample_idx]

        # Adds/Connects all subplots
        figure.add_subplot(rows, cols, i)

        # Add labels an description
        plt.title(str(label))
        plt.axis("off")
        plt.imshow(torch.tensor(img).squeeze(), cmap="gray")
    plt.show()    

def load_data_mnist():
    # Concatenate training and testing
    X = np.concatenate([train_X, test_X], axis=0).reshape((70000, 28*28))
    y = np.append(train_y, test_y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    #*********** Split the data set into training and testing**********#
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.2,train_size=0.8)
    # Converting to torch
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test  = Variable(torch.from_numpy(X_test)).float()
    y_test  = Variable(torch.from_numpy(y_test)).long()

    return X_train, y_train, X_test, y_test

    

def load_data_iris():
    X = iris.data
    y = iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #*********** Split the data set into training and testing**********#
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size=0.2,train_size=0.8)

    X_scaled = scaler.fit_transform(X)

    # Split Data into training and validation set
    # Converting to torch
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test  = Variable(torch.from_numpy(X_test)).float()
    y_test  = Variable(torch.from_numpy(y_test)).long()

    return X_train, y_train, X_test, y_test



# folder to load config file
CONFIG_PATH = "MNIST/"

# Function to load yaml configuration file
def load_config(config_name):

    if not config_name.endswith(".yaml"):
        raise ValueError("Invalid file type - must be a YAML file.")

    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


def plot_all_deterministic(L1Norm_all,train_loss_all, test_loss_all, accuracy, train_loss_loss_ws,l1_norm_values_ws):

    fig0 = plt.figure(2)

    L1Norm_start = L1Norm_all[0]
    L1Norm_pred_grad_all = L1Norm_all[1]
    L1Norm_corr_grad_all = L1Norm_all[2]
    L1Norm_pred_prox_all = L1Norm_all[3]
    L1Norm_corr_prox_all = L1Norm_all[4]

    train_loss_start = train_loss_all[0]
    train_loss_pred_grad_all = train_loss_all[1]
    train_loss_corr_grad_all = train_loss_all[2]
    train_loss_pred_prox_all = train_loss_all[3]
    train_loss_corr_prox_all = train_loss_all[4]

    test_loss_start = test_loss_all[0]
    test_loss_corr_grad_all = test_loss_all[1]
    test_loss_corr_prox_all = test_loss_all[2]

    train_acc_all = accuracy[0]
    test_acc_all = accuracy[1]

    plt.plot(train_loss_start, L1Norm_start, color = "gray")
    plt.scatter(train_loss_start[0::1000], L1Norm_start[0::1000], marker="x", color = "gray")

    good_pareto_loss = []
    good_pareto_L1 = []
    good_pareto_test_loss = []

    for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_) in zip(L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1])
        plt.plot(train_loss_pred_, L1Norm_pred_, color = "orange")
        plt.plot(train_loss_corr_, L1Norm_corr_, color = "blue")

    good_pareto_loss.append(train_loss_start[-1])
    good_pareto_test_loss.append(test_loss_start[-1])
    good_pareto_L1.append(L1Norm_start[-1])

    for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_) in zip(L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1])    
        plt.plot(train_loss_pred_, L1Norm_pred_, color = "red")
        plt.plot(train_loss_corr_, L1Norm_corr_, color = "green")


    plt.plot(train_loss_start, L1Norm_start, color = "gray", label = "start")
    plt.plot(train_loss_pred_grad_all[0], L1Norm_pred_grad_all[0], color = "orange", label = "predictor grad")
    plt.plot(train_loss_corr_grad_all[0], L1Norm_corr_grad_all[0], color = "blue", label = "corrector")
    plt.plot(train_loss_pred_prox_all[0], L1Norm_pred_prox_all[0], color = "red", label = "predictor prox")
    plt.plot(train_loss_corr_prox_all[0], L1Norm_corr_prox_all[0], color = "green", label = "corrector")   
    plt.ylabel('L1 Norm',fontsize = 15, fontweight='bold')
    plt.xlabel('loss',fontsize = 15, fontweight='bold')
    plt.title("Predictor-Corrector Deterministic", color="brown", fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend()
    plt.savefig("images/Continuation_deterministic", dpi='figure', format=None, metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()  


    plt.plot(good_pareto_loss, good_pareto_L1,linestyle = '-',marker='o', color= "black", label = "CM_train" )
    plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= "yellow", label = "CM_test" )
    plt.plot(train_loss_loss_ws, l1_norm_values_ws,"-o", color= "blue", label = "WS_train" )
    plt.ylabel("l1 norm")
    plt.xlabel("loss")
    plt.title ('Continuation Pareto front Deterministic', color="red", fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend()
    plt.savefig("MNIST/images/All_Paretofront_Deterministic", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)

    plt.show()


    plt.plot(train_acc_all, good_pareto_L1,"-o", color= "green", label = "CM_train_acc" )
    plt.plot(test_acc_all, good_pareto_L1,"-o", color= "brown", label = "CM_test_acc" )
    plt.ylabel("l1 norm")
    plt.xlabel("accuracy")
    plt.title ('Accuray plot for Cont. Pareto front Deterministic')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend()
    plt.savefig("MNIST/images/Deter_Accuracy_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()


def plot_all_stochastic(L1Norm_all_cm,train_loss_all_cm,test_loss_all_cm, accuracy_cm,
                        train_loss_values_ws,l1_norm_values_ws,test_loss_values_ws,
                        train_acc_all_ws,test_acc_all_ws):
    # Plot the Pareto front
    L1Norm_start = L1Norm_all_cm[0]
    L1Norm_pred_grad_all = L1Norm_all_cm[1]
    L1Norm_corr_grad_all = L1Norm_all_cm[2]
    L1Norm_pred_prox_all = L1Norm_all_cm[3]
    L1Norm_corr_prox_all = L1Norm_all_cm[4]

    train_loss_start = train_loss_all_cm[0]
    train_loss_pred_grad_all = train_loss_all_cm[1]
    train_loss_corr_grad_all = train_loss_all_cm[2]
    train_loss_pred_prox_all = train_loss_all_cm[3]
    train_loss_corr_prox_all = train_loss_all_cm[4]

    test_loss_start = test_loss_all_cm[0]
    test_loss_corr_grad_all = test_loss_all_cm[1]
    test_loss_corr_prox_all = test_loss_all_cm[2]

    train_acc_all = accuracy_cm[0]
    test_acc_all = accuracy_cm[1]

    good_pareto_loss = []
    good_pareto_L1 = []
    good_pareto_test_loss = []


    plt.plot(train_loss_start, L1Norm_start, color = "gray")
    plt.scatter(train_loss_start[0::1000], L1Norm_start[0::1000], marker="x", color = "gray")

    for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, test_loss_corr_grad_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1])
        good_pareto_test_loss.append(test_loss_corr_[-1])   
        plt.plot(train_loss_pred_, L1Norm_pred_, color = "orange")
        plt.plot(train_loss_corr_, L1Norm_corr_, color = "blue")     

    good_pareto_loss.append(train_loss_start[-1])
    good_pareto_test_loss.append(test_loss_start[-1])
    good_pareto_L1.append(L1Norm_start[-1])

    for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_,test_loss_corr_) in zip(L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, test_loss_corr_prox_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1]) 
        good_pareto_test_loss.append(test_loss_corr_[-1])   
        plt.plot(train_loss_pred_, L1Norm_pred_, color = "red")
        plt.plot(train_loss_corr_, L1Norm_corr_, color = "green")  



    #**************Predictor-corrector step*********************#
    plt.plot(train_loss_start, L1Norm_start, color = "gray", label = "start")
    plt.plot(train_loss_pred_grad_all[0], L1Norm_pred_grad_all[0], color = "orange", label = "predictor grad")
    plt.plot(train_loss_corr_grad_all[0], L1Norm_corr_grad_all[0], color = "blue", label = "corrector")
    plt.plot(train_loss_pred_prox_all[0], L1Norm_pred_prox_all[0], color = "red", label = "predictor prox")
    plt.plot(train_loss_corr_prox_all[0], L1Norm_corr_prox_all[0], color = "green", label = "corrector")
    plt.ylabel('L1 Norm', fontsize = 15, fontweight='bold')
    plt.xlabel('loss', fontsize = 15, fontweight='bold')
    plt.title("Predictor-Corrector Stochastic", fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend()
    plt.savefig("MNIST/images/Continuation_stochastic", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()  
        

    plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= "black", label = "train (CM)" , linewidth = 4, markersize= 8 )
    plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= "red", label = "test (CM)" , linewidth = 4, markersize= 8 )
    plt.plot(train_loss_values_ws,l1_norm_values_ws , '--o', color= "black", label='train (WS)', linewidth = 4, markersize=8 )
    plt.plot(test_loss_values_ws,l1_norm_values_ws, '--o', color= "red", label='test (WS)', linewidth = 4, markersize= 8)
    plt.ylabel("L1 norm", fontsize = 20, fontweight='bold')
    plt.xlabel("loss", fontsize = 20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title ('All Pareto front', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("MNIST/images/All_Paretofront_Stochastic", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 


    plt.plot( good_pareto_L1, train_acc_all,"-o", color= "black", label = "train acc (CM)", linewidth = 4, markersize= 8 )
    plt.plot( good_pareto_L1, test_acc_all,"-o",color= "red", label = "test acc (CM)", linewidth = 4, markersize= 8 )
    plt.plot( l1_norm_values_ws,train_acc_all_ws, "--o", color= "black", label = "train acc (WS)", linewidth = 4, markersize= 8 )
    plt.plot( l1_norm_values_ws, test_acc_all_ws,"--o", color= "red", label = "test acc (WS)", linewidth = 4, markersize= 8 )
    plt.ylabel("accuracy", fontsize = 20, fontweight='bold')
    plt.xlabel("L1 norm", fontsize = 20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title ('Accuray plot stochastic', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("MNIST/images/All_Accuracy_Plot", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()  