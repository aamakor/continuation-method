import pickle
import matplotlib.pyplot as plt
import os


cmyk_black = (0, 0, 0, 1)
cmyk_red = (1, 0, 0, 1)
cmyk_blue = (0,0, 1, 1)
cmyk_yellow = (1,1,0.2,1)
cmyk_purple = (1, 0, 1, 1)
cmyk_orange = (1, 0.5, 0, 1)
cmyk_green = (0, 0.5, 0, 1)
cmyk_gray = (0, 0, 0, 0.4)

    
directory = os.path.join(os.getcwd(), 'Results')

# **********Plot for stochastic setting predictor-corrector method (CM) CIFAR10 dataset****************
def plot_cifar10_cm():

    """Performs the plotting of the pareto front and accuracy plot for the predictor-corrector method (CM)
    """

    directory_names = [dir for dir in os.walk(directory)]
    directory_names = [dir[0] for dir in directory_names[1::2]]

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

        train_acc_all = accuracy[0]
        test_acc_all = accuracy[1]

        good_pareto_loss = [] #training loss
        good_pareto_L1 = [] # L1 norm
        good_pareto_test_loss = [] #testing loss

        train_pareto =[]
        train_L1 = []

        for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, test_loss_corr_grad_all):
            good_pareto_loss.append(train_loss_corr_[-1])
            good_pareto_L1.append(L1Norm_corr_[-1])
            good_pareto_test_loss.append(test_loss_corr_[-1]) 
            train_pareto.append( train_loss_pred_[-1])  
            train_L1.append(L1Norm_pred_[-1])
            
        good_pareto_loss.append(train_loss_start[-1])
        good_pareto_test_loss.append(test_loss_start[-1])
        good_pareto_L1.append(L1Norm_start[-1])



        for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_,test_loss_corr_) in zip(L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, test_loss_corr_prox_all):
            good_pareto_loss.append(train_loss_corr_[-1])
            good_pareto_L1.append(L1Norm_corr_[-1]) 
            good_pareto_test_loss.append(test_loss_corr_[-1]) 
            train_pareto.append( train_loss_pred_[-1])  
            train_L1.append(L1Norm_pred_[-1])

    # Plot the data as a line with markers
    plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= cmyk_black, label = "train" , linewidth = 4, markersize= 8)
    plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= cmyk_red, label = "test" , linewidth = 4, markersize= 8 )

    # Plot the intersection point with a different color
    plt.plot(train_loss_start[-1], L1Norm_start[-1], color =cmyk_blue, marker = 'o', markersize = 8, label = "initial point")
    plt.plot(test_loss_start[-1], L1Norm_start[-1], color =cmyk_blue, marker = 'o', markersize = 8)
    # Add labels and title
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('Continuation Pareto front Stochastic Cifar10', fontweight = 'bold', fontsize = 15)
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/Stoch_Cifar10", dpi='figure', format=None, metadata=None,
                bbox_inches="tight", pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()

    plt.plot( good_pareto_L1,train_acc_all,"-o", color= cmyk_black, label = "train acc", linewidth = 4, markersize= 8 )
    plt.plot( good_pareto_L1,test_acc_all, "-o", color= cmyk_red, label = "test acc", linewidth = 4, markersize= 8 )
    plt.ylabel("accuracy", fontsize = 20)
    plt.xlabel("$\\ell^1$ norm", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('Accuracy plot for CM Cifar10', fontweight = 'bold', fontsize = 15)
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/Accuracy_Cifar10", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()




def plot_cifar10_ws():

    """Performs the plotting of the pareto front and accuracy plot for the weighted sum method (WS)
    """
    # ******Reading file from the directory***********#
    directory_names = [dir for dir in os.walk(directory)]
    directory_names_ws = [dir[0] for dir in directory_names[2::]]
    print(directory_names_ws)

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

    plt.plot(train_loss_values_ws,l1_norm_values_ws , 'X', color= cmyk_black, label='train (WS)', linewidth = 4, markersize=10 )
    plt.plot(test_loss_values_ws,l1_norm_values_ws, 'X', color= cmyk_red, label='test (WS)', linewidth = 4, markersize= 8)
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title ('Weighted sum Pareto front stochastic Cifar10', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/WS_Paretofront_cifar10", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 


    plt.plot( l1_norm_values_ws,train_acc_all_ws, "X", color= cmyk_black, label = "train acc (WS)", linewidth = 4, markersize= 10 )
    plt.plot( l1_norm_values_ws, test_acc_all_ws,"X", color= cmyk_red, label = "test acc (WS)", linewidth = 4, markersize= 8 )
    plt.ylabel("accuracy", fontsize = 20)
    plt.xlabel("$\\ell^1$ norm", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title ('Accuracy plot for WS Cifar10', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/WS_Accuracy_cifar10", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 
    plt.close()  





# **********Plot for comparison between predictor-corrector method and weighted sum method stochastic setting for MNIST dataset****************
def plot_cifar10_cmws():

    """Performs the plotting of the comparison plots for the CM and WS (i.e., Pareto front and accuracy plot).
    """

    directory_names = [dir for dir in os.walk(directory)]
    directory_names_cm = [dir[0] for dir in directory_names[1::2]]
    directory_names_ws = [dir[0] for dir in directory_names[2::]]
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

    plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= cmyk_black, label = "train (CM)" , linewidth = 4, markersize= 10 )
    plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= cmyk_red, label = "test (CM)" , linewidth = 4, markersize= 10 )
    plt.plot(train_loss_values_ws,l1_norm_values_ws , 'X', color= cmyk_black, label='train (WS)', linewidth = 4, markersize=10 , markeredgewidth=2)
    plt.plot(test_loss_values_ws,l1_norm_values_ws, 'X', color= cmyk_red, label='test (WS)', linewidth = 4, markersize= 10)
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('All Pareto front cifar10', fontweight='bold', fontsize = 15)
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/All_Paretofront_CIFAR10", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 


    plt.plot( good_pareto_L1, train_acc_all,"-o", color= cmyk_black, label = "train acc (CM)", linewidth = 4, markersize= 10 )
    plt.plot( good_pareto_L1, test_acc_all,"-o",color= cmyk_red, label = "test acc (CM)", linewidth = 4, markersize= 10 )
    plt.plot( l1_norm_values_ws,train_acc_all_ws, "X", color= cmyk_black, label = "train acc (WS)", linewidth = 4, markersize= 10, markeredgewidth=2 )
    plt.plot( l1_norm_values_ws, test_acc_all_ws,"X", color= cmyk_red, label = "test acc (WS)", linewidth = 4, markersize= 10 )
    plt.ylabel("accuracy", fontsize = 20)
    plt.xlabel("$\\ell^1$ norm", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title ('All Accuracy plot cifar10', fontweight='bold', fontsize = 15)
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/All_Accuracy_CIFAR10", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()

plot_cifar10_cmws()
plot_cifar10_cm()
plot_cifar10_ws()