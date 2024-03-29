import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter
from matplotlib.markers import MarkerStyle


# Define CMYK colors used as specified by the aaai2024
cmyk_black = (0, 0, 0, 1)
cmyk_red = (1, 0, 0, 1)
cmyk_blue = (0,0, 1, 1)
cmyk_yellow = (1,1,0.2,1)
cmyk_purple = (1, 0, 1, 1)
cmyk_orange = (1, 0.5, 0, 1)
cmyk_green = (0, 0.5, 0, 1)
cmyk_gray = (0, 0, 0, 0.4)

# Get the directory of the current script
current_script_directory = os.path.dirname(__file__)

# Specify the relative path to the target file
target_folder = "../predictor-corrector/Results"

# Construct the relative path to the target file
directory = os.path.join(current_script_directory, target_folder)


def plot_dt_cmea():
    
    """Performs the plotting of the pareto front
    for the evolutionary algorthm (EA) for the Iris dataset
    """
    #X_pareto_front = np.loadtxt('Results/iris_X_pareto_front.txt')
    F_pareto_front = np.loadtxt('EA/Results/iris_F_pareto_front.txt')
    directory_names = [dir for dir in os.walk(directory)]
    #print(directory_names)
    directory_names = [dir[0] for dir in directory_names[1:2:]]
    #print([dir for dir in directory_names])

    for directory_name in directory_names:

        L1Norm_path = os.path.join(directory_name, f"L1NormAll")
        TrainLoss_path = os.path.join(directory_name, f"TrainLoss")
        TestLoss_path = os.path.join(directory_name, f"TestLoss")
        accuracy_path = os.path.join(directory_name, f"accuracy")

        with open (L1Norm_path,'rb') as pick:
            L1Norm_all = pickle.load(pick)

        with open (TrainLoss_path,'rb') as pick:
            train_loss_all = pickle.load(pick)

        with open (TestLoss_path,'rb') as pick:
            test_loss_all = pickle.load(pick)

        with open (accuracy_path,'rb') as pick:
            accuracy = pickle.load(pick)

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

        good_pareto_loss = [] #training loss
        good_pareto_L1 = [] # L1 norm
        good_pareto_test_loss = [] #testing loss

        for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, test_loss_corr_grad_all):
            good_pareto_loss.append(train_loss_corr_[-1])
            good_pareto_L1.append(L1Norm_corr_[-1])
            good_pareto_test_loss.append(test_loss_corr_[-1])
            #plt.plot(train_loss_pred_, L1Norm_pred_, color = "orange")
            #plt.plot(train_loss_corr_, L1Norm_corr_, color = "blue")
            
          

        good_pareto_loss.append(train_loss_start[-1])
        good_pareto_test_loss.append(test_loss_start[-1])
        good_pareto_L1.append(L1Norm_start[-1])

        for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_,test_loss_corr_) in zip(L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, test_loss_corr_prox_all):
            good_pareto_loss.append(train_loss_corr_[-1])
            good_pareto_L1.append(L1Norm_corr_[-1]) 
            good_pareto_test_loss.append(test_loss_corr_[-1])
         
    # Plot the data as a line with markers
    plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= cmyk_black, label = "train(CM)" , linewidth = 4, markersize= 8)
    plt.plot(sorted(F_pareto_front[:, 0], reverse=True), sorted(F_pareto_front[:, 1]),"-o", color= cmyk_blue, label = "train (NSGA-II)" , linewidth = 4, markersize= 8)

    # Plot the intersection point with a different color
    #plt.plot(good_pareto_loss[intersection_index], good_pareto_L1[intersection_index], 'o', color=cmyk_red, markersize=8, label='initial point')
    #plt.plot(good_pareto_test_loss[intersection_index_test], good_pareto_L1[intersection_index_test], 'o', color=cmyk_red, markersize=8)
    # Add labels and title
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title ('EA Pareto front Deterministic Iris', fontweight = 'bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("EA/images/DeterEA_Continuation_Pareto_front_Ir", dpi='figure', format=None, metadata=None,
                bbox_inches="tight", pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)

    plt.show()









def plot_dt_cmea_mnist():

    """Performs the plotting of the pareto front for the 
    evolutionary algorithm (EA) for the MNIST dataset
    """
    #X_pareto_front = np.loadtxt('Results/iris_X_pareto_front.txt')
    F_pareto_front = np.loadtxt('EA/Results/mnist_F_pareto_front.txt')   

    # Plot the data as a line with markers
    #plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= cmyk_black, label = "train(CM)" , linewidth = 4, markersize= 8)
    plt.plot(sorted(F_pareto_front[:, 0], reverse=True), sorted(F_pareto_front[:, 1]),"-o", color= cmyk_blue, label = "train (NSGA-II)" , linewidth = 4, markersize= 8)
   

    # Plot the intersection point with a different color
    #plt.plot(good_pareto_loss[intersection_index], good_pareto_L1[intersection_index], 'o', color=cmyk_red, markersize=8, label='initial point')
    #plt.plot(good_pareto_test_loss[intersection_index_test], good_pareto_L1[intersection_index_test], 'o', color=cmyk_red, markersize=8)
    # Add labels and title
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks#(fontsize=15)
    plt.yticks#(fontsize=15)
    #plt.title ('EA Pareto front Deterministic Iris', fontweight = 'bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("EA/images/StochEA_Continuation_Pareto_front_Ir", dpi='figure', format=None, metadata=None,
                bbox_inches="tight", pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)

    plt.show()




def plot_comp_cmwsea():

    """Performs the plotting of the comparison plots for the CM, 
    EA and WS (i.e., Pareto front training dataset stochastic setting).
    """
    #X_pareto_front = np.loadtxt('evolutionary-algo/Results/mnist_X_pareto_front.txt')
    F_pareto_front = np.loadtxt('EA/Results/mnist_F_pareto_front.txt')
    directory_names = [dir for dir in os.walk(directory)]
    directory_names_cm = [dir[0] for dir in directory_names[2::3]]
    directory_names_ws = [dir[0] for dir in directory_names[3::]]
    #print([dir for dir in directory_names])

    for directory_name in directory_names_cm:
        L1Norm_path = os.path.join(directory_name, f"L1NormAll")
        TrainLoss_path = os.path.join(directory_name, f"TrainLoss")
        TestLoss_path = os.path.join(directory_name, f"TestLoss")
        accuracy_path = os.path.join(directory_name, f"accuracy")

        with open (L1Norm_path,'rb') as pick:
            L1Norm_all = pickle.load(pick)
        with open (TrainLoss_path,'rb') as pick:
            train_loss_all = pickle.load(pick)
        with open (TestLoss_path,'rb') as pick:
            test_loss_all = pickle.load(pick)
        with open (accuracy_path,'rb') as pick:
            accuracy = pickle.load(pick)

        L1Norm_start = L1Norm_all[0]
        L1Norm_corr_grad_all = L1Norm_all[2]
        L1Norm_corr_prox_all = L1Norm_all[4]

        train_loss_start = train_loss_all[0]
        train_loss_corr_grad_all = train_loss_all[2]
        train_loss_corr_prox_all = train_loss_all[4]

        test_loss_start = test_loss_all[0]
        test_loss_corr_grad_all = test_loss_all[1]
        test_loss_corr_prox_all = test_loss_all[2]

        train_acc_all = accuracy[0]
        test_acc_all = accuracy[1]


    for directory_name in directory_names_ws:
        TrainLoss_path_ws = os.path.join(directory_name, "TrainLoss")
        L1Norm_path_ws = os.path.join(directory_name, "L1NormAll")
        TestLoss_path_ws = os.path.join(directory_name, "TestLoss")
        TrainAccuracy_path_ws = os.path.join(directory_name, "TrainAccuracy")
        TestAccuracy_path_ws = os.path.join(directory_name, "TestAccuracy")

        with open (TrainLoss_path_ws,'rb') as pick:
            train_loss_values_ws = pickle.load(pick)
        with open (L1Norm_path_ws,'rb') as pick:
            l1_norm_values_ws = pickle.load(pick)
        with open (TestLoss_path_ws,'rb') as pick:
            test_loss_values_ws = pickle.load(pick)
        with open (TrainAccuracy_path_ws,'rb') as pick:
            train_acc_all_ws= pickle.load(pick)
        with open (TestAccuracy_path_ws,'rb') as pick:
            test_acc_all_ws = pickle.load(pick)    

    good_pareto_loss = []
    good_pareto_L1 = []
    good_pareto_test_loss = []

    for (L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_corr_grad_all, train_loss_corr_grad_all, test_loss_corr_grad_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1])
        good_pareto_test_loss.append(test_loss_corr_[-1])  

    good_pareto_loss.append(train_loss_start[-1])
    good_pareto_test_loss.append(test_loss_start[-1])
    good_pareto_L1.append(L1Norm_start[-1])

    for (L1Norm_corr_, train_loss_corr_,test_loss_corr_) in zip(L1Norm_corr_prox_all, train_loss_corr_prox_all, test_loss_corr_prox_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1]) 
        good_pareto_test_loss.append(test_loss_corr_[-1])   

    plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= cmyk_black, label = "train (CM)" , linewidth = 4, markersize= 6 )
    #plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= cmyk_red, label = "test (CM)" , linewidth = 4, markersize= 10 )
    plt.plot(sorted(F_pareto_front[:, 0], reverse=True), sorted(F_pareto_front[:, 1]),"-o", color= cmyk_blue, label = "train (NSGA-II)" , linewidth = 4, markersize= 8)
    #plt.scatter(F_pareto_front[:, 0], F_pareto_front[:, 1], s=30, facecolors='none', edgecolors='blue', label = "train (EA)")
    plt.plot(train_loss_values_ws,l1_norm_values_ws , 'X', color= cmyk_red, label='train (WS)', linewidth = 4, markersize=6, markeredgewidth=2)
    #plt.plot(test_loss_values_ws,l1_norm_values_ws, 'X', color= cmyk_red, label='test (WS)', linewidth = 4, markersize= 10)

    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=15)
    #plt.title ('All Pareto front stochastic', fontweight='bold', fontsize = 15)
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    #plt.tight_layout()
    
    plt.savefig("EA/images/Allea_Paretofront_Stochastic", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 
    plt.close()



plot_dt_cmea()
plot_dt_cmea_mnist()
plot_comp_cmwsea()