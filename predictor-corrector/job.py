from DataLoader import load_data_mnist, load_data_iris
from continuationTest import continuation_methodMNIST_stochastic, continuation_methodIris
from weightedsumTest import weightedSum_methodMNIST_stochastic
from plotResults import plot_iris_reference_vs_cm, plot_sto_cm,plot_comp_cmws
from plotResultsCV import plot_cm_cv


#*********Load Data ***************#

#Uncomment to load data
#X_train,y_train, X_test,y_test = load_data_mnist()
#X_train_ir, y_train_ir, X_test_ir, y_test_ir = load_data_iris()


 #Uncomment to train and run algorithms
"""
# *********Deterministic training on iris data using predictor-corrector method ************

L1Norm_all,train_loss_all, test_loss_all, accuracy = continuation_methodIris(X_train_ir,y_train_ir, X_test_ir, y_test_ir)

#*************Stochastic training on mnist data  using predictor-corrector method***********#

L1Norm_all_cm,train_loss_all_cm,test_loss_all_cm, accuracy_cm = continuation_methodMNIST_stochastic(X_train,y_train, X_test,y_test)

#*************Stochastic training on mnist data using weighted sum method***********#
train_loss_values_ws,l1_norm_values_ws,test_loss_values_ws,train_acc_all_ws,test_acc_all_ws = weightedSum_methodMNIST_stochastic(X_train,y_train, X_test,y_test)

"""


#**************Plotting*****************#

#Run to plot
plot_iris_reference_vs_cm()
plot_sto_cm()
plot_comp_cmws()  

#cross validation
plot_cm_cv()


