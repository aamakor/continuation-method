import pickle
import matplotlib.pyplot as plt
import os


# Define CMYK colors used as specified by the aaai2024
cmyk_black = (0, 0, 0, 1)
cmyk_red = (1, 0, 0, 1)
cmyk_blue = (0,0, 1, 1)
cmyk_yellow = (1,1,0.2,1)
cmyk_purple = (1, 0, 1, 1)
cmyk_orange = (1, 0.5, 0, 1)
cmyk_green = (0, 0.5, 0, 1)
cmyk_gray = (0, 0, 0, 0.4)



directory = os.path.join(os.getcwd(), 'Results')

# **********Plot for deterministic setting predictor-corrector method (CM) for Iris dataset****************
def plot_dt_cm():

    """Performs the plotting of the pareto front and accuracy plot 
      for the predictor-corrector method (CM) for the Iris dataset
    """
    directory_names = [dir for dir in os.walk(directory)]
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
            #plt.plot(train_loss_pred_, L1Norm_pred_, color = "orange")
           # plt.plot(train_loss_corr_, L1Norm_corr_, color = "blue")


    #plt.plot(train_loss_start, L1Norm_start, color = "gray", label = "start")
    #plt.plot(train_loss_pred_grad_all[0], L1Norm_pred_grad_all[0], color = "orange", label = "predictor grad")
    #plt.plot(train_loss_corr_grad_all[0], L1Norm_corr_grad_all[0], color = "blue", label = "corrector")

    #plt.plot(train_loss_pred_prox_all[0], L1Norm_pred_prox_all[0], color = "red", label = "predictor prox")
    #plt.plot(train_loss_corr_prox_all[0], L1Norm_corr_prox_all[0], color = "green", label = "corrector")

    #plt.grid(visible=True, which='major', axis='both')
    #plt.legend()

    #plt.savefig("images/Deter_predictor_corrector_steps", dpi='figure', format=None, metadata=None,
            #bbox_inches=None, pad_inches=0.1,
            #facecolor='auto', edgecolor='auto',
           # backend=None)

    #plt.show()           

    # Plot the data as a line with markers
    plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= cmyk_black, label = "train" , linewidth = 4, markersize= 8)
    plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= cmyk_red, label = "test" , linewidth = 4, markersize= 8 )
    # Find the index of the intersection point 
    intersection_index = good_pareto_loss.index(train_loss_start[-1])
    intersection_index_test = good_pareto_test_loss.index(test_loss_start[-1])

    # Plot the intersection point with a different color
    plt.plot(good_pareto_loss[intersection_index], good_pareto_L1[intersection_index], 'o', color=cmyk_blue, markersize=8, label='initial point')
    plt.plot(good_pareto_test_loss[intersection_index_test], good_pareto_L1[intersection_index_test], 'o', color=cmyk_blue, markersize=8)
    # Add labels and title
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('CM Pareto front Deterministic Iris', fontweight = 'bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/Deter_Continuation_Pareto_front_Ir", dpi='figure', format=None, metadata=None,
                bbox_inches="tight", pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)

    plt.show()


def plot_reference_pf():
    """Performs the plotting of the reference Pareto front obtained from
       continuation method with random initial conditions
    """

    plt.ylabel("accuracy", fontsize = 20)
    plt.xlabel("$\\ell^1$ norm", fontsize = 20)
    plt.title("Reference Pareto front for Iris", fontweight = 'bold')

    directory = os.path.join(os.getcwd(), 'Results_reference')

    directory_names = [dir for dir in os.walk(directory)]
    directory_names = [dir[0] for dir in directory_names[1::]]

    for directory_name in directory_names:

        L1Norm_path = os.path.join(directory_name, f"L1Pareto")
        TrainLoss_path = os.path.join(directory_name, f"TrainPareto")

        with open (L1Norm_path,'rb') as pick:
            L1NormPareto = pickle.load(pick)

        with open (TrainLoss_path,'rb') as pick:
            train_loss_Pareto = pickle.load(pick)


        plt.plot(train_loss_Pareto, L1NormPareto, color = cmyk_blue, linewidth = 4)

    plt.grid(visible=True, which='major', axis='both')

    plt.savefig("images/reference_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)

    plt.show()
    plt.close()


def plot_iris_reference_vs_cm():
    
    """Performs the plotting of Figure 2 in the paper i.e.,  
       Pareto front approximation for the Iris dataset using 
       Algorithm 2 (red symbols) versus the reference Pareto
       front in “blue” (computed using the same algorithm with
       very small step sizes and many different initial conditions)
       with unscaled L1 norm
    """

    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.title("Reference vs Approximated Pareto front for Iris", fontweight = 'bold')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    directory = os.path.join(os.getcwd(), 'Results_reference')

    directory_names = [dir for dir in os.walk(directory)]
    directory_names = [dir[0] for dir in directory_names[1::]]

    for directory_name in directory_names:

        L1Norm_path = os.path.join(directory_name, f"L1Pareto")
        TrainLoss_path = os.path.join(directory_name, f"TrainPareto")

        with open (L1Norm_path,'rb') as pick:
            L1NormPareto = pickle.load(pick)

        with open (TrainLoss_path,'rb') as pick:
            train_loss_Pareto = pickle.load(pick)


        plt.plot(train_loss_Pareto, L1NormPareto, color = cmyk_blue, linewidth = 4)
    plt.plot(train_loss_Pareto, L1NormPareto, color = cmyk_blue, linewidth = 4, label = "reference Pareto front")
    
    directory = os.path.join(os.getcwd(), 'Results')

    directory_names = [dir for dir in os.walk(directory)]
    directory_names = [dir[0] for dir in directory_names[1:2:]]

    for directory_name in directory_names:

        L1Norm_path = os.path.join(directory_name, f"L1NormAll")
        TrainLoss_path = os.path.join(directory_name, f"TrainLoss")

        with open (L1Norm_path,'rb') as pick:
            L1Norm_all = pickle.load(pick)

        with open (TrainLoss_path,'rb') as pick:
            train_loss_all = pickle.load(pick)

        L1Norm_start = L1Norm_all[0]
        L1Norm_corr_grad_all = L1Norm_all[2]
        L1Norm_corr_prox_all = L1Norm_all[4]

        L1Norm_corr_grad_all = [l1[-1] for l1 in L1Norm_corr_grad_all]
        L1Norm_corr_prox_all = [l1[-1] for l1 in L1Norm_corr_prox_all]

        train_loss_start = train_loss_all[0]
        train_loss_corr_grad_all = train_loss_all[2]
        train_loss_corr_prox_all = train_loss_all[4]

        train_loss_corr_grad_all = [l1[-1] for l1 in train_loss_corr_grad_all]
        train_loss_corr_prox_all = [l1[-1] for l1 in train_loss_corr_prox_all]

        train_loss_start = train_loss_all[0]
        L1Norm_start = L1Norm_all[0]
    
        plt.plot(train_loss_start[-1], L1Norm_start[-1], color =cmyk_black, marker = 'o', markersize = 5,markeredgewidth=3, label = "initial point")

        plt.plot(train_loss_corr_grad_all, L1Norm_corr_grad_all, color = cmyk_red, marker = 'o', markersize = 6, linewidth = 0, label = "approximated point")
        plt.plot(train_loss_corr_prox_all, L1Norm_corr_prox_all, color = cmyk_red, marker = 'o', markersize = 6, linewidth = 0)

    plt.grid(visible=True, which='major', axis='both')
    plt.savefig("images/Reference_vs_Approximated_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches='tight', pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.show()
    plt.close()         



# **********Plot for stochastic setting predictor-corrector method (CM) MNIST dataset****************
def plot_sto_cm():

    """Performs the plotting of the pareto front and accuracy plot for the predictor-corrector method (CM)
    """

    directory_names = [dir for dir in os.walk(directory)]
    directory_names = [dir[0] for dir in directory_names[2::3]]
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

        train_acc_all = accuracy[0]
        test_acc_all = accuracy[1]

        good_pareto_loss = [] #training loss
        good_pareto_L1 = [] # L1 norm
        good_pareto_test_loss = [] #testing loss

        for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, test_loss_corr_grad_all):
            good_pareto_loss.append(train_loss_corr_[-1])
            good_pareto_L1.append(L1Norm_corr_[-1])
            good_pareto_test_loss.append(test_loss_corr_[-1])   


        good_pareto_loss.append(train_loss_start[-1])
        good_pareto_test_loss.append(test_loss_start[-1])
        good_pareto_L1.append(L1Norm_start[-1])

        for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_,test_loss_corr_) in zip(L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, test_loss_corr_prox_all):
            good_pareto_loss.append(train_loss_corr_[-1])
            good_pareto_L1.append(L1Norm_corr_[-1]) 
            good_pareto_test_loss.append(test_loss_corr_[-1])   
    
    # Plot the data as a line with markers
    plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= cmyk_black, label = "train" , linewidth = 4, markersize= 8)
    plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= cmyk_red, label = "test" , linewidth = 4, markersize= 8 )
    # Find the index of the intersection point 
    intersection_index = good_pareto_loss.index(train_loss_start[-1])
    intersection_index_test = good_pareto_test_loss.index(test_loss_start[-1])

    # Plot the intersection point with a different color
    plt.plot(good_pareto_loss[intersection_index], good_pareto_L1[intersection_index], 'o', color=cmyk_blue, markersize=8, label='initial point')
    plt.plot(good_pareto_test_loss[intersection_index_test], good_pareto_L1[intersection_index_test], 'o', color=cmyk_blue, markersize=8)
    # Add labels and title
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('Continuation Pareto front Stochastic', fontweight = 'bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/Stoch_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
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
    plt.title ('Accuracy plot for CM', fontweight = 'bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/Accuracy_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()
    plt.close() 



# **********Plot for stochastic setting WS for MNIST dataset****************
def plot_sto_ws():

    """Performs the plotting of the pareto front and accuracy plot for the weighted sum method (WS)
    """

    # ******Reading file from the directory***********#
    directory_names = [dir for dir in os.walk(directory)]
    directory_names_ws = [dir[0] for dir in directory_names[3::]]

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

    plt.plot(train_loss_values_ws,l1_norm_values_ws , '--o', color= cmyk_black, label='train (WS)', linewidth = 4, markersize=8 )
    plt.plot(test_loss_values_ws,l1_norm_values_ws, '--o', color= cmyk_red, label='test (WS)', linewidth = 4, markersize= 8)
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('Weighted sum Pareto front stochastic', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/WS_Paretofront_Stochastic", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 


    plt.plot( l1_norm_values_ws,train_acc_all_ws, "--o", color= cmyk_black, label = "train acc (WS)", linewidth = 4, markersize= 8 )
    plt.plot( l1_norm_values_ws, test_acc_all_ws,"--o", color= cmyk_red, label = "test acc (WS)", linewidth = 4, markersize= 8 )
    plt.ylabel("accuracy", fontsize = 20)
    plt.xlabel("$\\ell^1$ norm", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('Accuracy plot for WS', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/WS_Accuracy_Plot", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 
    plt.close()   


         
# **********Plot for comparison between predictor-corrector method and weighted sum method stochastic setting for MNIST dataset****************
def plot_comp_cmws():

    """Performs the plotting of the comparison plots for the CM and WS (i.e., Pareto front and accuracy plot).
    """
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

    plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= cmyk_black, label = "train (CM)" , linewidth = 4, markersize= 8 )
    plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= cmyk_red, label = "test (CM)" , linewidth = 4, markersize= 8 )
    plt.plot(train_loss_values_ws,l1_norm_values_ws , '--o', color= cmyk_black, label='train (WS)', linewidth = 4, markersize=8 )
    plt.plot(test_loss_values_ws,l1_norm_values_ws, '--o', color= cmyk_red, label='test (WS)', linewidth = 4, markersize= 8)
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('All Pareto front stochastic', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/All_Paretofront_Stochastic", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 


    plt.plot( good_pareto_L1, train_acc_all,"-o", color= cmyk_black, label = "train acc (CM)", linewidth = 4, markersize= 8 )
    plt.plot( good_pareto_L1, test_acc_all,"-o",color= cmyk_red, label = "test acc (CM)", linewidth = 4, markersize= 8 )
    plt.plot( l1_norm_values_ws,train_acc_all_ws, "--o", color= cmyk_black, label = "train acc (WS)", linewidth = 4, markersize= 8 )
    plt.plot( l1_norm_values_ws, test_acc_all_ws,"--o", color= cmyk_red, label = "test acc (WS)", linewidth = 4, markersize= 8 )
    plt.ylabel("accuracy", fontsize = 20)
    plt.xlabel("$\\ell^1$ norm", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title ('All Accuracy plot stochastic', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("images/All_Accuracy_Plot", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show()
    plt.close()
