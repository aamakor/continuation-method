import torch
import numpy as np
import tqdm
import torch.optim as optim
from helperFunctions import computeL1Norm,get_accuracy
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset


#**************predictor-corrector data training for stochastic setting cross validation ******************#
def train_test_data_st(type:str, X_train, y_train,X_test, y_test, model, optimizer, loss,fold, n_corr_first = 0 ,n_pareto_grad = 0, 
                     n_predictor_loss = 0, n_corrector_loss = 0, n_pareto_prox = 0, n_predictor = 0, n_corr_l1 = 0, batch_num = 0):

    """Performs a stochastic training and testing on the mnist dataset using Algorithm 2 as described in our
       paper "A multiobjective continuation method to compute the regularization path of deep neural networks"

    Parameters
        ----------
        type  : str could be 'first_iter', 'loss_iter', or 'l1norm_iter'
        X_train,y_train, X_test,y_test: The splitted training and testing sets of 80-20 ratio.
        model: the neural network model architectured to be used
        optimizer: The muliobjective proximal gradient optimizer
        loss: The loss objective function
        n_corr_first: number of training epochs for first run to obtain initial point on the front
        n_pareto_grad: number of points that hopefully belong to the pareto front (loss objective)
        n_predictor_loss: number of iterations for predictor step (gradient)
        n_corrector_loss: number of iterations for corrector step (Algorithm 1) for loss objective
        n_pareto_prox: number of points that hopefully belong to the pareto front (L1 norm objective)
        n_predictor:  number of iterations for predictor step (shrinkage) 
        n_corr_l1: Set number of training epochs for corrector step (Algorithm 1) for L1 objective function
        batch_num: mini-batch size

    Returns:
            'first_iter': 
                L1Norm_start: [Array of values for the L1 norm computed when finding the inital point on the front]
                train_loss_start: [Array of values for the training loss computed when finding the inital point on the front]
                test_loss_start: [Array of values for the test loss computed when finding the inital point on the front]
                train_acc_all: [List of training accuracy values inital iteration] 
                test_acc_all: [List of testing accuracy values inital iteration] 

            'loss_iter' $ 'l1norm_iter':
                L1Norm_pred_grad_all: [Array of values for the L1 norm computed during the predictor step]
                train_loss_pred_grad_all: [Array of values for the training loss computed during the predictor step]
                L1Norm_corr_grad_all: [Array of values for the L1 norm computed during the corrector step]
                train_loss_corr_grad_all: [Array of values for the training loss computed during the corrector step] 
                train_acc_all: [List of training accuracy values]  
                test_loss_corr_grad_all: [Array of values for the testing loss computed during the corrector step] 
                test_acc_all: [List of testing accuracy values] 

    """
    
    weight_length = n_pareto_prox + n_pareto_grad + 1 # for scaling

    #************ Predictor Optimizer **********
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001) 

    # ********** Accuracy**********
    train_acc_all = []
    test_acc_all = []
    
    # Cross-validation results storage
    fold_results = {}
   
    #***************************************************
    if type == 'first_iter':
        """
        for computing the initial point on the front using algorithm 1 only (Mulitobjective Proximal Gradient- MPG)
        """

        L1Norm_start = np.zeros((n_corr_first+1,))
        train_loss_start = np.zeros((n_corr_first+1,))
        

        # first run to get first point on the pareto front
        for epoch in tqdm.trange(n_corr_first):

            model.train()            
            # Shuffle the training data
            permutation = torch.randperm(X_train.shape[0])
            # Split the training data into mini-batches
            for i in range(0, X_train.shape[0], batch_num):
                indices = permutation[i:i+batch_num]
                batch_X, batch_y = X_train[indices], y_train[indices]

                optimizer.acceleration_step_st()
                # model prediction
                y_pred = model(batch_X)
                # compute loss
                Loss = loss(y_pred, batch_y)

                # store values for potential pareto point
                L1Norm_start[epoch] = computeL1Norm(model)/weight_length
                train_loss_start[epoch] = Loss.item()
                            
                # compute gradients
                optimizer.zero_grad()
                Loss.backward()

                # preform moo proximal gradient step
                optimizer.MOOproximalgradientstep_st()

         
        # model prediction
        y_pred = model(X_train)
        # compute loss
        Loss = loss(y_pred, y_train)
        # store values for potential pareto point
        L1Norm_start[n_corr_first] = computeL1Norm(model)/weight_length
        train_loss_start[n_corr_first] = Loss.item()
        # Compute training accuracy
        #train_acc = get_accuracy(model, X_train, y_train)
        #train_acc_all.append(train_acc)


        #### Testing ***************
        model.eval()
        test_loss_start = np.zeros((n_corr_first+1,))
        L1Norm_start_val = np.zeros((n_corr_first+1,))
        # Validation loop  
        with torch.no_grad():
            for i in range(0, X_test.shape[0], batch_num):
                batch_X, batch_y = X_test[i:i+batch_num], y_test[i:i+batch_num]
                outputs = model(batch_X)
                Loss_cv = loss(outputs , batch_y)
                L1Norm_start_val[epoch]  = computeL1Norm(model)/weight_length
                test_loss_start[epoch] = Loss_cv.item()
            
            y_pred = model(X_test)
            test_loss_start[n_corr_first] = loss(y_pred, y_test).item()
            L1Norm_start_val[n_corr_first]  = computeL1Norm(model)/weight_length
            # Compute testing accuracy
            test_acc = get_accuracy(model, X_test, y_test)
            test_acc_all.append(test_acc)
            
        fold_results[fold] = test_acc
        print(f"Fold [{fold + 1}] Accuracy: {test_acc:.2f}%")    

        return  L1Norm_start, train_loss_start,test_loss_start,L1Norm_start_val, train_acc_all, test_acc_all 

    elif type == 'loss_iter':
        """
        for loss objective function using equation 2 as predictor and Algorithm 1 as corrector step
        """

        L1Norm_pred_grad_all = []
        train_loss_pred_grad_all = []

        L1Norm_corr_grad_all = []
        train_loss_corr_grad_all = []
        L1Norm_corr_val_grad_all = []
        test_loss_corr_grad_all = []
        
        # continuation method
            # outer loop for predictor
            # inner loop for corrector
            # predictor = gradient step for loss
        for pareto in tqdm.trange(n_pareto_grad):

            L1Norm_pred = np.zeros((n_predictor_loss+1,))
            train_loss_pred = np.zeros((n_predictor_loss+1,))

            # perform a number of gradient steps for predictor step #

            for pred in range(n_predictor_loss):
                
                # Shuffle the training data
                permutation = torch.randperm(X_train.shape[0])

                # Split the training data into mini-batches
                for i in range(0, X_train.shape[0], batch_num):
                    indices = permutation[i:i+batch_num]
                    batch_X, batch_y = X_train[indices], y_train[indices]

                    # model prediction
                    y_pred = model(batch_X)
                    # compute loss
                    Loss = loss(y_pred, batch_y)
                    # store values for potential pareto point
                    L1Norm_pred[pred] = computeL1Norm(model)/weight_length
                    train_loss_pred[pred] = Loss.item()                 
                    # compute gradient
                    optimizer_adam.zero_grad()
                    Loss.backward()
                    # perform gradient step
                    optimizer_adam.step()

    
          
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)
            # store values for potential pareto point
            L1Norm_pred[n_predictor_loss] = computeL1Norm(model)/weight_length
            train_loss_pred[n_predictor_loss] = Loss.item()

            #L1Norm_pred_grad_all.append(L1Norm_pred.copy())
            #train_loss_pred_grad_all.append(train_loss_pred.copy())


            #*** Corrector step for Loss*************#
            L1Norm_corr = np.zeros((n_corrector_loss+1,))
            train_loss_corr = np.zeros((n_corrector_loss+1,))
           
            # inner loop for correction
            for corr in tqdm.trange(n_corrector_loss):
                model.train()
                # Shuffle the training data
                permutation = torch.randperm(X_train.shape[0])


                # Split the training data into mini-batches
                for i in range(0, X_train.shape[0], batch_num):
                    indices = permutation[i:i+batch_num]
                    batch_X, batch_y = X_train[indices], y_train[indices]

                    optimizer.acceleration_step_st()
                    # model prediction
                    y_pred = model(batch_X)
                    # compute loss
                    Loss = loss(y_pred, batch_y)
                            
                    # store values for potential pareto point
                    L1Norm_corr[corr] = computeL1Norm(model)/weight_length
                    train_loss_corr[corr] = Loss.item()

                    # compute gradient
                    optimizer.zero_grad()
                    Loss.backward()

                    # preform moo proximal gradient step
                    optimizer.MOOproximalgradientstep_st()

            # model prediction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)

            # store values for main pareto point
            L1Norm_corr[n_corrector_loss] = computeL1Norm(model)/weight_length
            train_loss_corr[n_corrector_loss] = Loss.item()
            # Compute training accuracy
            #train_acc = get_accuracy(model, X_train, y_train)
            #train_acc_all.append(train_acc)                

          
            #### Testing ***************
            model.eval()
            test_loss_corr = np.zeros((n_corrector_loss+1,))
            L1Norm_corr_val = np.zeros((n_corrector_loss+1,))
            # Validation loop  
            with torch.no_grad():
                for i in range(0, X_test.shape[0], batch_num):
                    batch_X, batch_y = X_test[i:i+batch_num], y_test[i:i+batch_num]
                    outputs = model(batch_X)
                    Loss_cv = loss(outputs , batch_y)
                    L1Norm_corr_val[corr]  = computeL1Norm(model)/weight_length
                    test_loss_corr[corr] = Loss_cv.item()
                
                y_pred = model(X_test)
                test_loss_corr[n_corrector_loss] = loss(y_pred, y_test).item()
                L1Norm_corr_val[n_corrector_loss]  = computeL1Norm(model)/weight_length
                # Compute testing accuracy
                test_acc = get_accuracy(model, X_test, y_test)
                test_acc_all.append(test_acc)
                
            fold_results[fold] = test_acc
            print(f"Fold [{fold + 1}] Accuracy: {test_acc:.2f}%")    

            L1Norm_corr_grad_all.append(L1Norm_corr.copy())
            train_loss_corr_grad_all.append(train_loss_corr.copy())
            L1Norm_corr_val_grad_all.append(L1Norm_corr_val.copy())
            test_loss_corr_grad_all.append(test_loss_corr.copy())

        
        test_loss_corr_grad_all = test_loss_corr_grad_all[::-1]
        test_acc_all = test_acc_all[::-1]
        train_acc_all = train_acc_all[::-1]
        L1Norm_corr_grad_all = L1Norm_corr_grad_all[::-1]
        train_loss_corr_grad_all =train_loss_corr_grad_all[::-1]

        return L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, train_acc_all, test_loss_corr_grad_all,L1Norm_corr_val_grad_all, test_acc_all 

    elif type == 'l1norm_iter':
        """
        for L1 norm objective function using equation 3 as predictor step and Algorithm 1 as corrector step
        """
        L1Norm_pred_prox_all = []
        train_loss_pred_prox_all = []

        L1Norm_corr_prox_all = []
        train_loss_corr_prox_all = []
        L1Norm_corr_val_prox_all = []
        test_loss_corr_prox_all = []

        # continuation method
        # outer loop for predictor
        # inner loop for corrector
        # predictor = shrinkage step for L1-Norm
        for pareto in tqdm.trange(n_pareto_prox):

            L1Norm_pred = np.zeros((n_predictor+1,))
            train_loss_pred = np.zeros((n_predictor+1,))
    
            # perform a number of gradient steps for prediction
            
            for pred in range(n_predictor):

                # model prediction
                y_pred = model(X_train)
                # compute loss
                Loss = loss(y_pred, y_train)        
                # store values for potential pareto point
                L1Norm_pred[pred] = computeL1Norm(model)/weight_length
                train_loss_pred[pred] = Loss.item()
                #compute gradient
                optimizer.zero_grad()
                Loss.backward()
                # perform gradient step
                optimizer.shrinkage()
      
            # model prediction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)
            # store values for potential pareto point
            L1Norm_pred[n_predictor] = computeL1Norm(model)/weight_length
            train_loss_pred[n_predictor] = Loss.item()

            L1Norm_pred_prox_all.append(L1Norm_pred.copy())
            train_loss_pred_prox_all.append(train_loss_pred.copy())


            L1Norm_corr = np.zeros((n_corr_l1+1,))
            train_loss_corr = np.zeros((n_corr_l1+1,))

            # inner loop for correction
            for corr in tqdm.trange(n_corr_l1):

                # Shuffle the training data
                permutation = torch.randperm(X_train.shape[0])

                # Split the training data into mini-batches
                for i in range(0, X_train.shape[0], batch_num):
                    indices = permutation[i:i+batch_num]
                    batch_X, batch_y = X_train[indices], y_train[indices]

                    optimizer.acceleration_step_st()
                    # model prediction
                    y_pred = model(batch_X)
                    # compute loss
                    Loss = loss(y_pred, batch_y)                   
                    # store values for potential pareto point
                    L1Norm_corr[corr] = computeL1Norm(model)/weight_length
                    train_loss_corr[corr] = Loss.item()
                    # compute gradient
                    optimizer.zero_grad()
                    Loss.backward()
                    # preform moo proximal gradient step
                    optimizer.MOOproximalgradientstep_st()
 
            # model prediction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)
            # store values for potential pareto point
            L1Norm_corr[n_corr_l1] = computeL1Norm(model)/weight_length
            train_loss_corr[n_corr_l1] = Loss.item()
            #train_acc = get_accuracy(model, X_train, y_train)
            #train_acc_all.append(train_acc)
     
            #### Testing ***************
            model.eval()
            test_loss_corr = np.zeros((n_corr_l1+1,))
            L1Norm_corr_val = np.zeros((n_corr_l1+1,))
            # Validation loop  
            with torch.no_grad():
                for i in range(0, X_test.shape[0], batch_num):
                    batch_X, batch_y = X_test[i:i+batch_num], y_test[i:i+batch_num]
                    outputs = model(batch_X)
                    Loss_cv = loss(outputs , batch_y)
                    L1Norm_corr_val[corr]  = computeL1Norm(model)/weight_length
                    test_loss_corr[corr] = Loss_cv.item()
                
                y_pred = model(X_test)
                test_loss_corr[n_corr_l1] = loss(y_pred, y_test).item()
                L1Norm_corr_val[n_corr_l1]  = computeL1Norm(model)/weight_length
                # Compute testing accuracy
                test_acc = get_accuracy(model, X_test, y_test)
                test_acc_all.append(test_acc)
                
            fold_results[fold] = test_acc
            print(f"Fold [{fold + 1}] Accuracy: {test_acc:.2f}%") 
            

            L1Norm_corr_prox_all.append(L1Norm_corr.copy())
            train_loss_corr_prox_all.append(train_loss_corr.copy())
            L1Norm_corr_val_prox_all.append(L1Norm_corr_val.copy())
            test_loss_corr_prox_all.append(test_loss_corr.copy())

        return L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, train_acc_all, test_loss_corr_prox_all,L1Norm_corr_val_prox_all, test_acc_all 

    else:
        raise ValueError(f"Invalid type value: {type}. Must be one of 'first_iter', 'loss_iter', or 'l1norm_iter'.") 
                 
        
