import pickle
import matplotlib.pyplot as plt
import os


directory = os.path.join(os.getcwd(), 'predictor-corrector/Results_CV/')



# **********Plot for stochastic setting predictor-corrector method (CM) MNIST dataset Cross Validation****************
def plot_cm_cv():

    """Performs the plotting of the pareto front and accuracy plot for the predictor-corrector method (CM)
    """

        
    L1Norm_path = os.path.join(directory, "L1NormAll")
    TrainLoss_path = os.path.join(directory, "TrainLoss")
    TestLoss_path = os.path.join(directory, "TestLoss")
    L1Normtest_path = os.path.join(directory, "L1NormtestAll")
    accuracy_path = os.path.join(directory, "accuracy")

    with open (L1Norm_path,'rb') as pick:
        L1Norm_all_fold = pickle.load(pick)

    with open (TrainLoss_path,'rb') as pick:
        train_loss_all_fold = pickle.load(pick)

    with open (TestLoss_path,'rb') as pick:
        test_loss_all_fold = pickle.load(pick)
        
    with open (L1Normtest_path,'rb') as pick:
        l1_Norm_test_all_fold = pickle.load(pick)

    with open (accuracy_path,'rb') as pick:
        accuracy_fold = pickle.load(pick)
        
    
    L1Norm_all1,L1Norm_all2,L1Norm_all3,L1Norm_all4,L1Norm_all5 = L1Norm_all_fold[0],L1Norm_all_fold[1],L1Norm_all_fold[2],L1Norm_all_fold[3],L1Norm_all_fold[4]
    train_loss_all1,train_loss_all2,train_loss_all3,train_loss_all4,train_loss_all5 = train_loss_all_fold[0],train_loss_all_fold[1],train_loss_all_fold[2],train_loss_all_fold[3],train_loss_all_fold[4]
    test_loss_all1,test_loss_all2,test_loss_all3,test_loss_all4,test_loss_all5 = test_loss_all_fold[0],test_loss_all_fold[1],test_loss_all_fold[2],test_loss_all_fold[3],test_loss_all_fold[4]
    L1Norm_allv1,L1Norm_allv2,L1Norm_allv3,L1Norm_allv4,L1Norm_allv5 = l1_Norm_test_all_fold[0],l1_Norm_test_all_fold[1],l1_Norm_test_all_fold[2],l1_Norm_test_all_fold[3],l1_Norm_test_all_fold[4]


    L1Norm_start1,L1Norm_corr_grad_all1,L1Norm_corr_prox_all1 = L1Norm_all1[0],L1Norm_all1[2],L1Norm_all1[4]
    L1Norm_start2,L1Norm_corr_grad_all2,L1Norm_corr_prox_all2 = L1Norm_all2[0],L1Norm_all2[2], L1Norm_all2[4]
    L1Norm_start3,L1Norm_corr_grad_all3,L1Norm_corr_prox_all3 = L1Norm_all3[0],L1Norm_all3[2],L1Norm_all3[4]
    L1Norm_start4,L1Norm_corr_grad_all4 , L1Norm_corr_prox_all4 = L1Norm_all4[0],L1Norm_all4[2], L1Norm_all4[4]
    L1Norm_start5,L1Norm_corr_grad_all5,L1Norm_corr_prox_all5 = L1Norm_all5[0],L1Norm_all5[2],L1Norm_all5[4]

    train_loss_start1,train_loss_corr_grad_all1,train_loss_corr_prox_all1 = train_loss_all1[0],train_loss_all1[2],train_loss_all1[4]
    train_loss_start2,train_loss_corr_grad_all2,train_loss_corr_prox_all2 = train_loss_all2[0],train_loss_all2[2], train_loss_all2[4]
    train_loss_start3,train_loss_corr_grad_all3,train_loss_corr_prox_all3 = train_loss_all3[0],train_loss_all3[2],train_loss_all3[4]
    train_loss_start4,train_loss_corr_grad_all4,train_loss_corr_prox_all4  = train_loss_all4[0],train_loss_all4[2],train_loss_all4[4]
    train_loss_start5,train_loss_corr_grad_all5,train_loss_corr_prox_all5 = train_loss_all5[0],train_loss_all5[2],train_loss_all5[4]
    
    
    test_loss_start1,test_loss_corr_grad_all1,test_loss_corr_prox_all1 = test_loss_all1[0],test_loss_all1[1],test_loss_all1[2]
    test_loss_start2,test_loss_corr_grad_all2,test_loss_corr_prox_all2  = test_loss_all2[0], test_loss_all2[1],test_loss_all2[2]
    test_loss_start3,test_loss_corr_grad_all3,test_loss_corr_prox_all3 = test_loss_all3[0],test_loss_all3[1],test_loss_all3[2]
    test_loss_start4,test_loss_corr_grad_all4,test_loss_corr_prox_all4 = test_loss_all4[0],test_loss_all4[1], test_loss_all4[2]
    test_loss_start5,test_loss_corr_grad_all5,test_loss_corr_prox_all5 = test_loss_all5[0],test_loss_all5[1],test_loss_all5[2]


    l1_Norm_valid_start1,l1_Norm_valid_grad_all1,l1_Norm_valid_prox_all1  = L1Norm_allv1[0],L1Norm_allv1[1],L1Norm_allv1[2]
    l1_Norm_valid_start2,l1_Norm_valid_grad_all2,l1_Norm_valid_prox_all2 = L1Norm_allv2[0],L1Norm_allv2[1],L1Norm_allv2[2]
    l1_Norm_valid_start3,l1_Norm_valid_grad_all3,l1_Norm_valid_prox_all3 = L1Norm_allv3[0],L1Norm_allv3[1],L1Norm_allv3[2]
    l1_Norm_valid_start4, l1_Norm_valid_grad_all4,l1_Norm_valid_prox_all4 = L1Norm_allv4[0],L1Norm_allv4[1],L1Norm_allv4[2]
    l1_Norm_valid_start5,l1_Norm_valid_grad_all5,l1_Norm_valid_prox_all5 = L1Norm_allv5[0],L1Norm_allv5[1],L1Norm_allv5[2]
    
    

    good_pareto_train_loss1, good_pareto_train_L11,good_pareto_train_loss2,good_pareto_train_L12, good_pareto_train_loss3, \
        good_pareto_train_L13,good_pareto_train_loss4,good_pareto_train_L14,good_pareto_train_loss5,good_pareto_train_L15 = [],[],[],[],[],[],[],[],[],[]


    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_grad_all1, train_loss_corr_grad_all1):
        good_pareto_train_loss1.append(train_loss_corr_[-1])
        good_pareto_train_L11.append(L1Norm_corr_[-1])
        
    good_pareto_train_loss1.append(train_loss_start1[-1])
    good_pareto_train_L11.append(L1Norm_start1[-1])

    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_prox_all1, train_loss_corr_prox_all1):
        good_pareto_train_loss1.append(train_loss_corr_[-1])
        good_pareto_train_L11.append(L1Norm_corr_[-1]) 
        
    #fold2 

    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_grad_all2, train_loss_corr_grad_all2):
        good_pareto_train_loss2.append(train_loss_corr_[-1])
        good_pareto_train_L12.append(L1Norm_corr_[-1])
        
    good_pareto_train_loss2.append(train_loss_start2[-1])
    good_pareto_train_L12.append(L1Norm_start2[-1])

    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_prox_all2, train_loss_corr_prox_all2):
        good_pareto_train_loss2.append(train_loss_corr_[-1])
        good_pareto_train_L12.append(L1Norm_corr_[-1]) 
    
    #fold3 

    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_grad_all3, train_loss_corr_grad_all3):
        good_pareto_train_loss3.append(train_loss_corr_[-1])
        good_pareto_train_L13.append(L1Norm_corr_[-1])
        
    good_pareto_train_loss3.append(train_loss_start3[-1])
    good_pareto_train_L13.append(L1Norm_start3[-1])

    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_prox_all3, train_loss_corr_prox_all3):
        good_pareto_train_loss3.append(train_loss_corr_[-1])
        good_pareto_train_L13.append(L1Norm_corr_[-1]) 
    

    #fold4
    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_grad_all4, train_loss_corr_grad_all4):
        good_pareto_train_loss4.append(train_loss_corr_[-1])
        good_pareto_train_L14.append(L1Norm_corr_[-1])
        
    good_pareto_train_loss4.append(train_loss_start4[-1])
    good_pareto_train_L14.append(L1Norm_start4[-1])

    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_prox_all4, train_loss_corr_prox_all4):
        good_pareto_train_loss4.append(train_loss_corr_[-1])
        good_pareto_train_L14.append(L1Norm_corr_[-1]) 

    #fold5 
    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_grad_all5, train_loss_corr_grad_all5):
        good_pareto_train_loss5.append(train_loss_corr_[-1])
        good_pareto_train_L15.append(L1Norm_corr_[-1])
        
    good_pareto_train_loss5.append(train_loss_start5[-1])
    good_pareto_train_L15.append(L1Norm_start5[-1])

    for (L1Norm_corr_, train_loss_corr_) in zip(L1Norm_corr_prox_all5, train_loss_corr_prox_all5):
        good_pareto_train_loss5.append(train_loss_corr_[-1])
        good_pareto_train_L15.append(L1Norm_corr_[-1]) 
        
        
    
    good_pareto_L1_val1,good_pareto_test_loss1,good_pareto_L1_val2,good_pareto_test_loss2,good_pareto_L1_val3, \
    good_pareto_test_loss3,good_pareto_L1_val4 ,good_pareto_test_loss4,good_pareto_L1_val5, good_pareto_test_loss5 = [],[],[],[],[],[],[],[],[],[]

    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_grad_all1, test_loss_corr_grad_all1):
        good_pareto_L1_val1.append(L1Norm_corr_[-1])
        good_pareto_test_loss1.append(test_loss_corr_[-1])  


    good_pareto_test_loss1.append(test_loss_start1[-1])
    good_pareto_L1_val1.append(l1_Norm_valid_start1[-1])

    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_prox_all1,  test_loss_corr_prox_all1):
        good_pareto_L1_val1.append(L1Norm_corr_[-1]) 
        good_pareto_test_loss1.append(test_loss_corr_[-1])   

    #fold 2    
        
    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_grad_all2, test_loss_corr_grad_all2):
        good_pareto_L1_val2.append(L1Norm_corr_[-1])
        good_pareto_test_loss2.append(test_loss_corr_[-1])  

    good_pareto_test_loss2.append(test_loss_start2[-1])
    good_pareto_L1_val2.append(l1_Norm_valid_start2[-1])

    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_prox_all2,  test_loss_corr_prox_all2):
        good_pareto_L1_val2.append(L1Norm_corr_[-1]) 
        good_pareto_test_loss2.append(test_loss_corr_[-1])   
        
    #fold 3
    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_grad_all3, test_loss_corr_grad_all3):
        good_pareto_L1_val3.append(L1Norm_corr_[-1])
        good_pareto_test_loss3.append(test_loss_corr_[-1])  

    good_pareto_test_loss3.append(test_loss_start3[-1])
    good_pareto_L1_val3.append(l1_Norm_valid_start3[-1])

    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_prox_all3,  test_loss_corr_prox_all3):
        good_pareto_L1_val3.append(L1Norm_corr_[-1]) 
        good_pareto_test_loss3.append(test_loss_corr_[-1])   

    #fold 4
    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_grad_all4, test_loss_corr_grad_all4):
        good_pareto_L1_val4.append(L1Norm_corr_[-1])
        good_pareto_test_loss4.append(test_loss_corr_[-1])  

    good_pareto_test_loss4.append(test_loss_start4[-1])
    good_pareto_L1_val4.append(l1_Norm_valid_start4[-1])

    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_prox_all4,  test_loss_corr_prox_all4):
        good_pareto_L1_val4.append(L1Norm_corr_[-1]) 
        good_pareto_test_loss4.append(test_loss_corr_[-1])   

    #fold 5
        
    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_grad_all5, test_loss_corr_grad_all5):
        good_pareto_L1_val5.append(L1Norm_corr_[-1])
        good_pareto_test_loss5.append(test_loss_corr_[-1])  

    good_pareto_test_loss5.append(test_loss_start5[-1])
    good_pareto_L1_val5.append(l1_Norm_valid_start5[-1])

    for (L1Norm_corr_, test_loss_corr_) in zip(l1_Norm_valid_prox_all5,  test_loss_corr_prox_all5):
        good_pareto_L1_val5.append(L1Norm_corr_[-1]) 
        good_pareto_test_loss5.append(test_loss_corr_[-1])       

            

    plt.plot(good_pareto_train_loss1, good_pareto_train_L11,"o", label = "fold1" )
    plt.plot(good_pareto_train_loss2, good_pareto_train_L12,"o", label = "fold2" )
    plt.plot(good_pareto_train_loss3, good_pareto_train_L13,"o", label = "fold3"  )
    plt.plot(good_pareto_train_loss4, good_pareto_train_L14,"o", label = "fold4"  )
    plt.plot(good_pareto_train_loss5, good_pareto_train_L15,"o", label = "fold5" )
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title ('Pareto front for 5 fold CV train set', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("predictor-corrector/images/CVTR_All_Paretofront_Stochastic", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 
    
    


    plt.plot(good_pareto_test_loss1, good_pareto_L1_val1,"o", label = "fold1" )
    plt.plot(good_pareto_test_loss2, good_pareto_L1_val2,"o", label = "fold2" )
    plt.plot(good_pareto_test_loss3, good_pareto_L1_val3,"o", label = "fold3"  )
    plt.plot(good_pareto_test_loss4, good_pareto_L1_val4,"o", label = "fold4"  )
    plt.plot(good_pareto_test_loss5, good_pareto_L1_val5,"o", label = "fold5" )
    plt.ylabel("$\\ell^1$ norm", fontsize = 20)
    plt.xlabel("loss", fontsize = 20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.title ('Pareto front for 5 fold CV Validation set', fontweight='bold')
    plt.grid(visible=True, which='major', axis='both')
    plt.legend(prop = { "size": 15 })
    plt.tight_layout()
    plt.savefig("predictor-corrector/images/CVVAL_All_Paretofront_Stochastic", dpi='figure', format=None, metadata=None,
                bbox_inches='tight', pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None)
    plt.show() 
    plt.close() 



plot_cm_cv()