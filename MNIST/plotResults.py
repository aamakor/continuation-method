import pickle
import matplotlib.pyplot as plt
import os
from importlib import resources
import io




plt.ylabel("l1 norm")
plt.xlabel("loss")

directory = 'MNIST\\Results'

directory_names = [dir for dir in os.walk(directory)]
#print([dir[0] for dir in directory_names[1:2:]])

"""

# **********Plot for Iris data****************

#****** Uncomment for Iris******

directory_names = [dir[0] for dir in directory_names[2::3]]

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

    plt.plot(train_loss_start, L1Norm_start, color = "gray")
    plt.scatter(train_loss_start[0::1000], L1Norm_start[0::1000], marker="x", color = "gray")

    good_pareto_loss = []
    good_pareto_L1 = []
    good_pareto_test_loss = []

    for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, test_loss_corr_grad_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1])
        good_pareto_test_loss.append(test_loss_corr_[-1])
        plt.plot(train_loss_pred_, L1Norm_pred_, color = "orange")
        plt.plot(train_loss_corr_, L1Norm_corr_, color = "blue")

    good_pareto_loss.append(train_loss_start[-1])
    good_pareto_test_loss.append(test_loss_start[-1])
    good_pareto_L1.append(L1Norm_start[-1])

    for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, test_loss_corr_prox_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1])
        good_pareto_test_loss.append(test_loss_corr_[-1])    
        plt.plot(train_loss_pred_, L1Norm_pred_, color = "red")
        plt.plot(train_loss_corr_, L1Norm_corr_, color = "green")

plt.plot(train_loss_start, L1Norm_start, color = "gray", label = "start")
plt.plot(train_loss_pred_grad_all[0], L1Norm_pred_grad_all[0], color = "orange", label = "predictor grad")
plt.plot(train_loss_corr_grad_all[0], L1Norm_corr_grad_all[0], color = "blue", label = "corrector")
plt.plot(train_loss_pred_prox_all[0], L1Norm_pred_prox_all[0], color = "red", label = "predictor prox")
plt.plot(train_loss_corr_prox_all[0], L1Norm_corr_prox_all[0], color = "green", label = "corrector")   
plt.ylabel('L1 Norm',fontsize = 15, fontweight='bold')
plt.xlabel('loss',fontsize = 15, fontweight='bold')
plt.title("Predictor-Corrector Deterministic", color="black", fontweight='bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.savefig("MNIST/images/Ir_Continuation_deterministic", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show()  


plt.plot(good_pareto_loss, good_pareto_L1,linestyle = '-',marker='o', color= "black", label = "CM_train" )
plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= "yellow", label = "CM_test" )
plt.ylabel("l1 norm", fontsize = 15, fontweight='bold')
plt.xlabel("loss", fontsize = 15, fontweight='bold')
plt.title ('Continuation Pareto front Deterministic', color="black", fontweight='bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.savefig("MNIST/images/Ir_Deter_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show() 


plt.plot( good_pareto_L1, train_acc_all, "-o", color= "green", label = "CM_train_acc" )
plt.plot( good_pareto_L1, test_acc_all, "-o", color= "brown", label = "CM_test_acc" )
plt.ylabel("l1 norm", fontsize = 15, fontweight='bold')
plt.xlabel("accuracy", fontsize = 15, fontweight='bold')
plt.title ('Accuray plot for Cont. Pareto front Deterministic', fontweight = 'bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.savefig("MNIST/images/Ir_Deter_Accuracy_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show()
plt.close()
"""

"""
# **********Plot for deterministic setting ****************

#****** Uncomment for deterministic setting******
directory_names = [dir[0] for dir in directory_names[1:2:]]
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

    plt.plot(train_loss_start, L1Norm_start, color = "gray")
    plt.scatter(train_loss_start[0::1000], L1Norm_start[0::1000], marker="x", color = "gray")

    good_pareto_loss = []
    good_pareto_L1 = []
    good_pareto_test_loss = []

    for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, test_loss_corr_grad_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1])
        good_pareto_test_loss.append(test_loss_corr_[-1])
        plt.plot(train_loss_pred_, L1Norm_pred_, color = "orange")
        plt.plot(train_loss_corr_, L1Norm_corr_, color = "blue")

    good_pareto_loss.append(train_loss_start[-1])
    good_pareto_test_loss.append(test_loss_start[-1])
    good_pareto_L1.append(L1Norm_start[-1])

    for (L1Norm_pred_, train_loss_pred_, L1Norm_corr_, train_loss_corr_, test_loss_corr_) in zip(L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, test_loss_corr_prox_all):
        good_pareto_loss.append(train_loss_corr_[-1])
        good_pareto_L1.append(L1Norm_corr_[-1])
        good_pareto_test_loss.append(test_loss_corr_[-1])    
        plt.plot(train_loss_pred_, L1Norm_pred_, color = "red")
        plt.plot(train_loss_corr_, L1Norm_corr_, color = "green")

plt.plot(train_loss_start, L1Norm_start, color = "gray", label = "start")
plt.plot(train_loss_pred_grad_all[0], L1Norm_pred_grad_all[0], color = "orange", label = "predictor grad")
plt.plot(train_loss_corr_grad_all[0], L1Norm_corr_grad_all[0], color = "blue", label = "corrector")
plt.plot(train_loss_pred_prox_all[0], L1Norm_pred_prox_all[0], color = "red", label = "predictor prox")
plt.plot(train_loss_corr_prox_all[0], L1Norm_corr_prox_all[0], color = "green", label = "corrector")   
plt.ylabel('l1 Norm',fontsize = 15, fontweight='bold')
plt.xlabel('loss',fontsize = 15, fontweight='bold')
plt.title("Predictor-Corrector Deterministic", color="brown", fontweight='bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.savefig("MNIST/images/Continuation_deterministic", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show()  


plt.plot(good_pareto_loss, good_pareto_L1,linestyle = '-',marker='o', color= "black", label = "CM_train" )
plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= "yellow", label = "CM_test" )
plt.ylabel("l1 norm", fontsize = 15, fontweight='bold')
plt.xlabel("loss", fontsize = 15, fontweight='bold')
plt.title ('Continuation Pareto front Deterministic', color="red", fontweight='bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.savefig("MNIST/images/Deter_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show() 


plt.plot( good_pareto_L1, train_acc_all, "-o", color= "green", label = "CM_train_acc" )
plt.plot( good_pareto_L1, test_acc_all, "-o", color= "brown", label = "CM_test_acc" )
plt.ylabel("l1 norm", fontsize = 15, fontweight='bold')
plt.xlabel("accuracy", fontsize = 15, fontweight='bold')
plt.title ('Accuray plot for Cont. Pareto front Deterministic', fontweight = 'bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend()
plt.savefig("MNIST/images/Deter_Accuracy_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show()
plt.close()
"""


# **********Plot for stochastic setting ****************
directory_names = [dir[0] for dir in directory_names[3::]]

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
plt.title("Predictor-Corrector Stochastic", fontweight = 'bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend(prop = { "size": 15 })
plt.savefig("MNIST/images/Continuation_stochastic", dpi='figure', format=None, metadata=None,
            bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show()  

# Create a figure and axis object
#fig, ax = plt.subplots()
# Plot the data with conditional formatting for the marker color
for i in range(len(good_pareto_loss)):
    if good_pareto_loss[i] == train_loss_start[-1] and good_pareto_L1[i] == L1Norm_start[-1]:
        plt.plot(good_pareto_loss[i], good_pareto_L1[i], '-o', color='yellow', markersize=8)
    else:
        plt.plot(good_pareto_loss[i], good_pareto_L1[i], '-o', color='black', markersize=8)

# Plot the data as a line with markers
plt.plot(good_pareto_loss, good_pareto_L1,"-o", color= "black", label = "train" , linewidth = 4, markersize= 8)
plt.plot(good_pareto_test_loss, good_pareto_L1,"-o", color= "red", label = "test" , linewidth = 4, markersize= 8 )
# Find the index of the intersection point 
intersection_index = good_pareto_loss.index(train_loss_start[-1])
intersection_index_test = good_pareto_test_loss.index(test_loss_start[-1])

# Plot the intersection point with a different color
plt.plot(good_pareto_loss[intersection_index], good_pareto_L1[intersection_index], 'o', color='blue', markersize=8, label='initial point')
plt.plot(good_pareto_test_loss[intersection_index], good_pareto_L1[intersection_index], 'o', color='blue', markersize=8)
# Add labels and title
plt.ylabel("l1 norm",fontsize = 20, fontweight='bold')
plt.xlabel("loss", fontsize = 20, fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.title ('Continuation Pareto front Stochastic', fontweight = 'bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend(prop = { "size": 15 })
plt.tight_layout()
plt.savefig("MNIST/images/Stoch_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches="tight", pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)

plt.show() 

plt.plot( good_pareto_L1,train_acc_all, color= "black", label = "train acc", linewidth = 4 )
plt.plot( good_pareto_L1,test_acc_all, color= "red", label = "test acc", linewidth = 4 )
plt.ylabel("accuracy", fontsize = 20, fontweight='bold')
plt.xlabel("l1 norm", fontsize = 20, fontweight='bold')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.title ('Accuray plot for Cont. Pareto front', fontweight = 'bold')
plt.grid(visible=True, which='major', axis='both')
plt.legend(prop = { "size": 15 })
plt.tight_layout()
plt.savefig("MNIST/images/Accuracy_Continuation_Pareto_front", dpi='figure', format=None, metadata=None,
            bbox_inches='tight', pad_inches=0.1,
            facecolor='auto', edgecolor='auto',
            backend=None)
plt.show() 

#plt.close()

"""
from bokeh.plotting import figure, show
p = figure(title="Training and Testing accuracy",
           x_axis_label="l1 norm", y_axis_label="Accuracy")
#epochs_array = good_pareto_L1
p.line(good_pareto_L1, train_acc_all, legend_label="Training",
       color="blue", line_width=3)
p.line(good_pareto_L1, test_acc_all, legend_label="Testing",
       color="green",line_width=3)
p.legend.click_policy = "hide"
p.legend.location = 'bottom_right'
show(p)
"""