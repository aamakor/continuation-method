import torch
import torch.nn as nn
from functions import train_test_data_st,train_test_data_dt
from OwnDescent import OwnDescent
from DataLoader import load_data_mnist, load_data_iris, load_config
from helperFunctions import fixSparseStructure
import pickle
import os
import time




#*********Load mnist data ***************#

X_train_m,y_train_m, X_test_m,y_test_m = load_data_mnist()

#*********Load iris data ***************#
X_train_ir, y_train_ir, X_test_ir, y_test_ir = load_data_iris()


#***********Network for mnist dataset**************#
net_mnist = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )


#***********Network for iris data**************
net_iris= nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 4),
        nn.Tanh(),
        nn.Linear(4, 4),
        nn.Tanh(),
        nn.Linear(4, 3)
    )


#************************IRIS DATA SET *************************#
def continuation_methodIris(X_train_ir, y_train_ir, X_test_ir, y_test_ir):
    
    """Performs a deterministic training and testing on the iris dataset using the predictor-corrector algorithm.
        Args:
            X_train_m,y_train_m, X_test_m,y_test_m: The splitted iris training and testing sets of 80-20 ratio.
    """

    X_train, y_train, X_test, y_test = X_train_ir, y_train_ir, X_test_ir, y_test_ir
    start = time.time()
    # Setting configuriation using the yaml file
    config = load_config("configuration_iris.yaml")

    n_continuation = config["n_continuation"]


    for k_ in range(n_continuation):

        model = net_iris


        # Fix sparse neural network structure
        fixSparseStructure(model, 0.1)
        
        #**** Initialize parameters ****#
        params = model.parameters()
        learning_rate = 5e-2
        shrinkage_rate = 5e-3
        optimi = OwnDescent(params, lr=learning_rate, sr=shrinkage_rate, alpha = 4, eps = 1e-2)
        n_pgrad = config["n_pareto_grad"]
        n_pprox = config["n_pareto_prox"]
        n_pred = config["n_predictor"]
        n_cor = config["n_corr"]
        n_corfir = config["n_corr_first"]
        n_pred_los=  config["n_predictor_loss"]
        n_cor_los= config["n_corrector_loss"]
        loss = nn.CrossEntropyLoss()
        
        type1 = 'first_iter'
        type2 = 'loss_iter'
        type3 = 'l1norm_iter'

        L1Norm_start, train_loss_start, test_loss_start, train_acc_all0, test_acc_all0  = train_test_data_dt(type1, X_train, y_train, X_test, y_test, model=model, optimizer=optimi,
                                                                                                              loss = loss ,n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox, n_corr_first=n_corfir) 

        #*******Saving the weights to memory**********
        W_1_start = model[1].weight.clone().detach()
        b_1_start = model[1].bias.clone().detach()
        W_3_start = model[3].weight.clone().detach()
        b_3_start = model[3].bias.clone().detach()
        W_5_start = model[5].weight.clone().detach()
        b_5_start = model[5].bias.clone().detach()
        Weights_start = [W_1_start, b_1_start, W_3_start, b_3_start, W_5_start, b_5_start]
     
        
        L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, train_acc_all1, test_loss_corr_grad_all, test_acc_all1  = train_test_data_dt(type2, X_train, y_train,X_test, y_test,
                                                                                                                                                        model=model, optimizer=optimi, loss= loss ,n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox,
                                                                                                                                                       n_predictor_loss = n_pred_los,n_corrector_loss=n_cor_los)       
        
        W_1_pred_grad = model[1].weight.clone().detach()
        b_1_pred_grad = model[1].bias.clone().detach()
        W_3_pred_grad = model[3].weight.clone().detach()
        b_3_pred_grad = model[3].bias.clone().detach()
        W_5_pred_grad = model[5].weight.clone().detach()
        b_5_pred_grad = model[5].bias.clone().detach()

        Weights_pred_grad = [W_1_pred_grad, b_1_pred_grad, W_3_pred_grad, b_3_pred_grad, W_5_pred_grad, b_5_pred_grad]
            
        # set weights to weights that where computed from start iteration
        for group in optimi.param_groups:           
            with torch.no_grad():
                group['params'][0].copy_(nn.Parameter(W_1_start))
                group['params'][1].copy_(nn.Parameter(b_1_start))
                group['params'][2].copy_(nn.Parameter(W_3_start))
                group['params'][3].copy_(nn.Parameter(b_3_start))
                group['params'][4].copy_(nn.Parameter(W_5_start))
                group['params'][5].copy_(nn.Parameter(b_5_start))
       
        L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, train_acc_all2, test_loss_corr_prox_all, test_acc_all2  = train_test_data_dt(type3, X_train, y_train, X_test, y_test,
                                                                                                                                                            model=model, optimizer=optimi, loss= loss, n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox,
                                                                                                                                                           n_predictor=n_pred, n_corr_l1=n_cor)
        
        W_1_pred_prox = model[1].weight.clone().detach()
        b_1_pred_prox = model[1].bias.clone().detach()
        W_3_pred_prox = model[3].weight.clone().detach()
        b_3_pred_prox = model[3].bias.clone().detach()
        W_5_pred_prox = model[5].weight.clone().detach()
        b_5_pred_prox = model[5].bias.clone().detach()



        Weights_pred_prox = [W_1_pred_prox, b_1_pred_prox, W_3_pred_prox, b_3_pred_prox, W_5_pred_prox, b_5_pred_prox]
        
        
        train_acc_all = train_acc_all1 + train_acc_all0 + train_acc_all2
        test_acc_all = test_acc_all1 + test_acc_all0 + test_acc_all2
        
        Weights_all = [Weights_start, Weights_pred_grad, Weights_pred_prox]
        L1Norm_all = [L1Norm_start, L1Norm_pred_grad_all, L1Norm_corr_grad_all, L1Norm_pred_prox_all, L1Norm_corr_prox_all]
        train_loss_all = [train_loss_start, train_loss_pred_grad_all, train_loss_corr_grad_all, train_loss_pred_prox_all, train_loss_corr_prox_all]
        test_loss_all = [test_loss_start, test_loss_corr_grad_all, test_loss_corr_prox_all]
        accuracy = [train_acc_all,test_acc_all]

    end = time.time()


    #store values

    # Parent Directory path
    parent_dir = os.path.join(os.getcwd(), 'predictor-corrector/Results')

    directory_names = [dir for dir in os.walk(parent_dir)]
    directory_names = [dir[0] for dir in directory_names[1::]]
    L = len(directory_names) # L number of result files

    # Directory
    directory = "Results_cm_dt_iris"
    
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

    Weights_path = os.path.join(path, "Weights")    
    with open(Weights_path, "wb") as fp:   #Pickling
        pickle.dump(Weights_all, fp)       
    with open(accuracy_path, "wb") as fp:   #Pickling
        pickle.dump(accuracy, fp)

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
        f.write(f'Training loss after start iteration = {train_loss_start[-1]}\n')
        f.write(f'Test loss after start iteration = {test_loss_start[-1]}\n')
        f.write(f'L1 norm after start iteration = {L1Norm_start[-1]}\n')
        f.write(f'Training accuracy after start iteration = {train_acc_all0[-1]}\n')
        f.write(f'Testing accuracy after start iteration = {test_acc_all0[-1]}\n')
        f.write('\n')
        f.write(f'Training loss after grad continuation = {train_loss_corr_grad_all[-1][-1]}\n')
        f.write(f'Test loss after grad continuation = {test_loss_corr_grad_all[-1][-1]}\n')
        f.write(f'L1 norm after grad continuation = {L1Norm_corr_grad_all[-1][-1]}\n')
        f.write(f'Training accuracy after grad continuation = {train_acc_all1[-1]}\n')
        f.write(f'Testing accuracy after grad continuation = {test_acc_all1[-1]}\n')
        f.write('\n')
        f.write(f'Training loss after prox continuation = {train_loss_corr_prox_all[-1][-1]}\n')
        f.write(f'Test loss after prox continuation = {test_loss_corr_prox_all[-1][-1]}\n')
        f.write(f'L1 norm after prox continuation = {L1Norm_corr_prox_all[-1][-1]}\n')
        f.write(f'Training accuracy after prox continuation = {train_acc_all2[-1]}\n')
        f.write(f'Testing accuracy after prox continuation = {test_acc_all2[-1]}\n')
        f.write('\n')
        f.write(f'Total computation time for deterministic Train/Test = {end-start}\n')
        f.write('\n')

    print("finish")

    return L1Norm_all, train_loss_all, test_loss_all, accuracy



def continuation_methodMNIST_stochastic(X_train_m,y_train_m, X_test_m,y_test_m  ): 

    """Performs a stochastic training and testing on the mnist dataset using the predictor-corrector algorithm.
        Args:
            X_train_m,y_train_m, X_test_m,y_test_m: The splitted mnist training and testing sets of 80-20 ratio.
    """

    X_train,y_train, X_test,y_test = X_train_m,y_train_m, X_test_m,y_test_m 
    # Setting configuriation using the yaml file
    config = load_config("configuration.yaml")

    start = time.time()
    n_continuation = config["n_continuation"]
    
    

    for k_ in range(n_continuation):

        model = net_mnist

        #**** Initialize parameters ****#
        params = model.parameters()
        learning_rate = 1e-1
        shrinkage_rate = 5e-4
        optimi = OwnDescent(params, lr=learning_rate, sr=shrinkage_rate, alpha = 2, eps= 0)
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

        L1Norm_start, train_loss_start,test_loss_start, train_acc_all0, test_acc_all0   = train_test_data_st(type1, X_train, y_train, X_test, y_test, model=model, optimizer=optimi,
                                                                           loss = loss ,n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox, n_corr_first=n_corfir, batch_num =batch_num) 

        
        #*******Saving the weights to memory**********
        W_1_start = model[1].weight.clone().detach()
        b_1_start = model[1].bias.clone().detach()
        W_3_start = model[3].weight.clone().detach()
        b_3_start = model[3].bias.clone().detach()
        W_5_start = model[5].weight.clone().detach()
        b_5_start = model[5].bias.clone().detach()
        Weights_start = [W_1_start, b_1_start, W_3_start, b_3_start, W_5_start, b_5_start]
        
        L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, train_acc_all1, test_loss_corr_grad_all, test_acc_all1  = train_test_data_st(type2, X_train, y_train,X_test, y_test,
                                                                                                                                                        model=model, optimizer=optimi, loss= loss ,n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox,
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
       
        L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, train_acc_all2, test_loss_corr_prox_all, test_acc_all2  = train_test_data_st(type3, X_train, y_train,X_test, y_test,
                                                                                                                                                            model=model, optimizer=optimi, loss= loss, n_pareto_grad=n_pgrad, n_pareto_prox=n_pprox,
                                                                                                                                                            n_predictor=n_pred, n_corr_l1=n_cor, batch_num=batch_num)
        
        train_acc_all = train_acc_all1 + train_acc_all0 + train_acc_all2
        test_acc_all = test_acc_all1 + test_acc_all0 + test_acc_all2

        L1Norm_all = [L1Norm_start, L1Norm_pred_grad_all, L1Norm_corr_grad_all, L1Norm_pred_prox_all, L1Norm_corr_prox_all]
        train_loss_all = [train_loss_start, train_loss_pred_grad_all, train_loss_corr_grad_all, train_loss_pred_prox_all, train_loss_corr_prox_all]
        test_loss_all = [test_loss_start, test_loss_corr_grad_all, test_loss_corr_prox_all]
        accuracy = [train_acc_all,test_acc_all]

    end = time.time()
        # store values

    # Parent Directory path
    parent_dir = os.path.join(os.getcwd(), 'predictor-corrector/Results')

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
        f.write(f'Training loss after start iteration = {train_loss_start[-1]}\n')
        f.write(f'Test loss after start iteration = {test_loss_start[-1]}\n')
        f.write(f'L1 norm after start iteration = {L1Norm_start[-1]}\n')
        f.write(f'Training accuracy after start iteration = {train_acc_all0[-1]}\n')
        f.write(f'Testing accuracy after start iteration = {test_acc_all0[-1]}\n')
        f.write('\n')
        f.write(f'Training loss after grad continuation = {train_loss_corr_grad_all[-1][-1]}\n')
        f.write(f'Test loss after grad continuation = {test_loss_corr_grad_all[-1][-1]}\n')
        f.write(f'L1 norm after grad continuation = {L1Norm_corr_grad_all[-1][-1]}\n')
        f.write(f'Training accuracy after grad continuation = {train_acc_all1[-1]}\n')
        f.write(f'Testing accuracy after grad continuation = {test_acc_all1[-1]}\n')
        f.write('\n')
        f.write(f'Training loss after prox continuation = {train_loss_corr_prox_all[-1][-1]}\n')
        f.write(f'Test loss after prox continuation = {test_loss_corr_prox_all[-1][-1]}\n')
        f.write(f'L1 norm after prox continuation = {L1Norm_corr_prox_all[-1][-1]}\n')
        f.write(f'Training accuracy after prox continuation = {train_acc_all2[-1]}\n')
        f.write(f'Testing accuracy after prox continuation = {test_acc_all2[-1]}\n')
        f.write('\n')
        f.write(f'Total computation time for deterministic Train/Test = {end-start}\n')
        f.write('\n')

    print("finish")

    return L1Norm_all, train_loss_all, test_loss_all, accuracy


# *********Deterministic training on iris data ************

#L1Norm_all,train_loss_all, test_loss_all, accuracy = continuation_methodIris(X_train_ir,y_train_ir, X_test_ir, y_test_ir)

# **********Stochastic training on mnist data ****************

#L1Norm_all,train_loss_all, test_loss_all, accuracy = continuation_methodMNIST_stochastic(X_train_m,y_train_m, X_test_m,y_test_m )

