from .DataLoader import load_data_mnist, load_config , plot_all_deterministic, plot_all_stochastic
from .continuationTest import continuation_methodMNIST_stochastic
from .weightedsumTest import weightedSum_methodMNIST, weightedSum_methodMNIST_stochastic


#*********Load Data ***************#

X_train,y_train, X_test,y_test = load_data_mnist()


#*************Continuation_deterministic***********#

#L1Norm_all,train_loss_all, test_loss_all = continuation_methodMNIST(X_train,y_train,X_valid, y_valid , X_test,y_test)

#*************weightedSum_deterministic***********#
#train_loss_weight,l1_norm_weight = weightedSum_methodMNIST(X_train,y_train,X_valid, y_valid , X_test,y_test)


#plot_all_deterministic (L1Norm_all,train_loss_all, test_loss_all,train_loss_weight,l1_norm_weight)



#from pyinstrument import Profiler

#profiler = Profiler()
#profiler.start()

#*************Continuation_deterministic***********#

L1Norm_all_cm,train_loss_all_cm,test_loss_all_cm, accuracy_cm = continuation_methodMNIST_stochastic(X_train,y_train, X_test,y_test)

#*************weightedSum_deterministic***********#
train_loss_values_ws,l1_norm_values_ws,test_loss_values_ws,train_acc_all_ws,test_acc_all_ws = weightedSum_methodMNIST_stochastic(X_train,y_train, X_test,y_test)


plot_all_stochastic(L1Norm_all_cm,train_loss_all_cm,test_loss_all_cm, accuracy_cm,
                        train_loss_values_ws,l1_norm_values_ws,test_loss_values_ws,
                        train_acc_all_ws,test_acc_all_ws)

"""
from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

# code
profiler.stop()

print(profiler.output_text())
"""
#profiler.stop()

#print(profiler.output_text())