# Using cross validation for train CM##############

import torch
import torch.nn as nn
from  functionsCV import train_test_data_st
from OwnDescent import OwnDescent
from DataLoader import load_data_mnist
import pickle
import os
import time
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import io
import random
import numpy as np
import warnings
import yaml
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


CONFIG_PATH = "predictor-corrector/"

# Function to load yaml configuration file
def load_config(config_name):
    """Performs the loading of the configuration files. 
    """
    if not config_name.endswith(".yaml"):
        raise ValueError("Invalid file type - must be a YAML file.")

    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

#*********Load mnist data ***************#

X_train_m,y_train_m, X_test_m,y_test_m = load_data_mnist()
# Set up stratified k-fold cross-validation
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
targets = y_train_m.numpy()  # Labels for stratified splitting
# Cross-validation results storage
fold_results = {}

#***********Network for mnist dataset**************#
net_mnist = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

# Initialize the model
model = net_mnist#.state_dict()

# --- TRAINING LOGIC ---
# You would normally train the model here. For now, we skip training.
print("Model initialized.")

# Save the model to a memory buffer
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
print("Model saved to memory buffer.")


# Setting configuriation using the yaml file

config = load_config("configuration.yaml")

n_continuation = config["n_continuation"]
#model = net_mnist

#**** Initialize parameters ****#

n_pgrad = config["n_pareto_grad"]
n_pprox = config["n_pareto_prox"]
n_pred = config["n_predictor"]
n_cor = config["n_corr"]
n_corfir = config["n_corr_first"]
n_pred_los=  config["n_predictor_loss"]
n_cor_los= config["n_corrector_loss"]
batch_num = 5600
loss = nn.CrossEntropyLoss()

type1 = 'first_iter'
type2 = 'loss_iter'
type3 = 'l1norm_iter'

def continuation_methodMNIST_stochastic(X_train_m,y_train_m, X_test_m,y_test_m, fold,model): 

    """Performs a stochastic training and testing on the mnist dataset using the predictor-corrector algorithm.
        Args:
            X_train_m,y_train_m, X_test_m,y_test_m: The splitted mnist training and testing sets of 80-20 ratio.
    """
    
    params = model.parameters()
    learning_rate = 1e-1
    shrinkage_rate = 5e-4
    optimi = OwnDescent(params, lr=learning_rate, sr=shrinkage_rate, alpha = 2, eps= 0)

    X_train,y_train, X_test,y_test = X_train_m,y_train_m, X_test_m,y_test_m 

    start = time.time()
    for k_ in range(n_continuation):
        L1Norm_start, train_loss_start,test_loss_start,L1_Norm_valid_cv0 , train_acc_all0, test_acc_all0  = train_test_data_st(type1, X_train, y_train, X_test, y_test, model=model, optimizer=optimi,
                                                                           loss = loss ,fold= fold, n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox, n_corr_first=n_corfir, batch_num =batch_num) 

        
        #*******Saving the weights to memory**********
        W_1_start = model[1].weight.clone().detach()
        b_1_start = model[1].bias.clone().detach()
        W_3_start = model[3].weight.clone().detach()
        b_3_start = model[3].bias.clone().detach()
        W_5_start = model[5].weight.clone().detach()
        b_5_start = model[5].bias.clone().detach()
        Weights_start = [W_1_start, b_1_start, W_3_start, b_3_start, W_5_start, b_5_start]
        
        L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, train_acc_all1, test_loss_corr_grad_all,L1_Norm_valid_cv1, test_acc_all1 = train_test_data_st(type2, X_train, y_train,X_test, y_test,
                                                                                                                                                        model=model, optimizer=optimi, loss= loss ,fold= fold,n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox,
                                                                                                                                                       n_predictor_loss = n_pred_los,n_corrector_loss=n_cor_los,  batch_num=batch_num)       
        # set weights to weights that where computed from start iteration
        for group in optimi.param_groups:           
            with torch.no_grad():
                group['params'][0].copy_(nn.Parameter(W_1_start))
                group['params'][1].copy_(nn.Parameter(b_1_start))
                group['params'][2].copy_(nn.Parameter(W_3_start))
                group['params'][3].copy_(nn.Parameter(b_3_start))
                group['params'][4].copy_(nn.Parameter(W_5_start))
                group['params'][5].copy_(nn.Parameter(b_5_start))
       
        L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, train_acc_all2, test_loss_corr_prox_all,L1_Norm_valid_cv2, test_acc_all2   = train_test_data_st(type3, X_train, y_train,X_test, y_test,
                                                                                                                                                            model=model, optimizer=optimi, loss= loss,fold= fold, n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox,
                                                                                                                                                            n_predictor=n_pred, n_corr_l1=n_cor, batch_num=batch_num)
        
        train_acc_all = train_acc_all1 + train_acc_all0 + train_acc_all2
        test_acc_all = test_acc_all1 + test_acc_all0 + test_acc_all2

        L1Norm_all = [L1Norm_start, L1Norm_pred_grad_all, L1Norm_corr_grad_all, L1Norm_pred_prox_all, L1Norm_corr_prox_all]
        train_loss_all = [train_loss_start, train_loss_pred_grad_all, train_loss_corr_grad_all, train_loss_pred_prox_all, train_loss_corr_prox_all]
        test_loss_all = [test_loss_start, test_loss_corr_grad_all, test_loss_corr_prox_all]
        l1_Norm_test_all = [L1_Norm_valid_cv0, L1_Norm_valid_cv1, L1_Norm_valid_cv2]
        accuracy = [train_acc_all,test_acc_all]
        
    end = time.time()
 
    print(f'Total computation time for fold {fold} = {end-start}\n')
    return L1Norm_all, train_loss_all, test_loss_all, accuracy, l1_Norm_test_all


###***********Train a Stratified 5-fold cross validation

def cross_validation():

    L1Norm_all_fold,train_loss_all_fold, test_loss_all_fold, accuracy_fold, l1_Norm_test_all_fold = [],[],[],[],[]

    start = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_m, targets)):
        print(f"\nFold [{fold + 1}/{k_folds}]")
        
        # Load the model from the memory buffer
        buffer.seek(0)  # Reset buffer position to the beginning
        model_loaded = net_mnist  # Initialize a new model instance
        model_loaded.load_state_dict(torch.load(buffer))
        print("Model loaded from memory buffer.")

        
        # Create data subsets for the fold
        # Create train and validation splits
        X_fold_train, y_fold_train = X_train_m[train_idx], y_train_m[train_idx]
        X_fold_val, y_fold_val = X_train_m[val_idx], y_train_m[val_idx]
        
        
        L1Norm_all,train_loss_all, test_loss_all, accuracy,  l1_Norm_test_all= continuation_methodMNIST_stochastic(X_fold_train,y_fold_train, X_fold_val,y_fold_val, fold,model_loaded )
        
        
        L1Norm_all_fold.append(L1Norm_all),train_loss_all_fold.append(train_loss_all), test_loss_all_fold.append(test_loss_all), \
            accuracy_fold.append(accuracy), l1_Norm_test_all_fold.append(l1_Norm_test_all)
            
    end = time.time()
    # Parent Directory path
    parent_dir = os.path.join(os.getcwd(), 'predictor-corrector/Results_CV')

    #directory_names = [dir for dir in os.walk(parent_dir)]
    #directory_names = [dir[0] for dir in directory_names[1::]]
    #L = len(directory_names) # L number of result files

    # Directory
    #directory = "Results_cm_sto"
    
    # Path
    path = parent_dir #os.path.join(parent_dir, directory)

    # create directory in path
    if not os.path.exists(path):
        os.mkdir(path)
    
    L1Norm_path = os.path.join(path, "L1NormAll")
    TrainLoss_path = os.path.join(path, "TrainLoss")
    TestLoss_path = os.path.join(path, "TestLoss")
    L1Normtest_path = os.path.join(path, "L1NormtestAll")
    accuracy_path = os.path.join(path, "accuracy")
    
    with open(L1Norm_path, "wb") as fp:   #Pickling
        pickle.dump(L1Norm_all_fold, fp)
    with open(TrainLoss_path, "wb") as fp:   #Pickling
        pickle.dump(train_loss_all_fold, fp)
    with open(TestLoss_path, "wb") as fp:   #Pickling
        pickle.dump(test_loss_all_fold, fp)
    with open(L1Normtest_path, "wb") as fp:   #Pickling
        pickle.dump(l1_Norm_test_all_fold, fp)
        
    with open(accuracy_path, "wb") as fp:   #Pickling
        pickle.dump(accuracy_fold, fp)
    

    Text_file_path = os.path.join(path, "Info.txt")

    with open(Text_file_path, 'w') as f:
        
        f.write('# number of runs of continuation method runs\n')
        f.write(f'n_continuation = {n_continuation}\n')

        f.write('# number of points that hopefully belong to the pareto front for grad continuation\n')
        f.write(f'n_pareto = {n_pgrad}\n')
        
        f.write('# number of points that hopefully belong to the pareto front for prox continuation\n')
        f.write(f'n_pareto = {n_pprox}\n')

        f.write('# numper of iterations for prediction for loss\n')
        f.write(f'n_predictor = {n_pred_los}\n')

        f.write('# number of training epochs for corrector for loss\n')
        f.write(f'n_corr = {n_cor_los}\n')

        f.write('# numper of iterations for prediction for l1-norm\n')
        f.write(f'n_predictor = {n_pred}\n')

        f.write('# number of training epochs for corrector for l1-norm\n')
        f.write(f'n_corr = {n_cor}\n')

        f.write('# number of training epochs for first run\n')
        f.write(f'n_corr_first = {n_corfir}\n')


        f.write('\n')
        f.write(f'Total computation time for deterministic Train/Test = {end-start}\n')
        f.write('\n')

    print("finish")    
    
    return L1Norm_all_fold,train_loss_all_fold, test_loss_all_fold, accuracy_fold, l1_Norm_test_all_fold






# **********Cross Validation Stochastic training on mnist data ****************

L1Norm_all_fold,train_loss_all_fold, test_loss_all_fold, accuracy_fold, l1_Norm_test_all_fold = cross_validation()

