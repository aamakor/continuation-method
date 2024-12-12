import torch
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

# Load Iris Data
iris = load_iris()

# Load MNIST Data into np arrays
(train_X, train_y), (test_X, test_y) = mnist.load_data()


# Data Visualization
def visualize_training_data():
    """Performs a sample visualization of images of the mnist data. 
    """
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


def load_data_iris():
    """Performs the loading and splitting of  iris dataset into train and test sets
        to obtain same dataset for comparison with the reference Pareto front.
       Returns:
            X_train,y_train, X_test,y_test: The splitted training and testing sets of 80-20 ratio. 
    """
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']
    # Scale data to have mean 0 and variance 1 
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split the data set into training and testing
    N = len(X_scaled[:,1])
    P = np.loadtxt('predictor-corrector/perm.txt', dtype=int)
    X_scaled = np.array([X_scaled[i, :] for i in P])
    y = np.array([y[i] for i in P])
    I = range(0,N,5) # 80-20 ratio (train and test sets)
    I_c = [i for i in range(0,N) if i not in I]
    X_test = np.array([X_scaled[i, :] for i in I_c])
    X_train = np.array([X_scaled[i, :] for i in I])
    y_test = np.array([y[i] for i in I_c])
    y_train = np.array([y[i] for i in I])

    # Split Data into training and validation set
    # Converting to torch
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test  = Variable(torch.from_numpy(X_test)).float()
    y_test  = Variable(torch.from_numpy(y_test)).long()

    return X_train, y_train, X_test, y_test


def load_data_mnist():
    """Performs the loading and splitting of  mnist dataset into train and test sets.

       Returns:
            X_train,y_train, X_test,y_test: The splitted training and testing sets of 80-20 ratio. 
    """

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

# folder to load config file
CONFIG_PATH = "../predictor-corrector/"

# Function to load yaml configuration file
def load_config(config_name):
    """Performs the loading of the configuration files. 
    """
    if not config_name.endswith(".yaml"):
        raise ValueError("Invalid file type - must be a YAML file.")

    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

