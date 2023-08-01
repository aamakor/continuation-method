import torch
import numpy as np
import tqdm
import torch.optim as optim
import os
from .helperFunctions import computeL1Norm,get_accuracy



def train_test_data_dt(type:str, X_train, y_train, X_test, y_test, model, optimizer, loss, n_corr_first = 0 ,n_pareto_grad = 0, 
                       n_predictor_loss = 0, n_corrector_loss = 0, n_pareto_prox = 0, n_predictor = 0, n_corr_l1 = 0):

    
    #************ Predictor Optimizer **********
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001) 

    weight_length = n_pareto_prox + n_pareto_grad + 1

    #****************Accuracy*********************
    train_acc_all = []
    test_acc_all = []

    #***************************************************


    if type == 'first_iter':

        L1Norm_start = np.zeros((n_corr_first+1,))
        train_loss_start = np.zeros((n_corr_first+1,))
        test_loss_start = np.zeros((n_corr_first+1,))

        # first run to get first point on the pareto front
        for epoch in tqdm.trange(n_corr_first):

            model.train()  
            optimizer.acceleration_step()

            # model prediction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)

            # store values for potential pareto point
            L1Norm_start[epoch] = computeL1Norm(model)/weight_length
            train_loss_start[epoch] = Loss.item()
            # compute gradients
            optimizer.zero_grad()
            Loss.backward()
            # preform moo proximal gradient step
            optimizer.MOOproximalgradientstep()

        # model prediction
        y_pred = model(X_train)
        # compute loss
        Loss = loss(y_pred, y_train)
        # store values for potential pareto point
        L1Norm_start[n_corr_first] = computeL1Norm(model)/weight_length
        train_loss_start[n_corr_first] = Loss.item()
        train_acc = get_accuracy(model, X_train, y_train)
        train_acc_all.append(train_acc)
        
        #### Testing ***************
        model.eval()  
        with torch.no_grad():
            y_pred = model(X_test)
            test_loss_start[n_corr_first] = loss(y_pred, y_test).item()
            test_acc = get_accuracy(model, X_test, y_test)
            test_acc_all.append(test_acc)

        return  L1Norm_start, train_loss_start, test_loss_start,train_acc_all,test_acc_all
    

    elif type == 'loss_iter':

        L1Norm_pred_grad_all = []
        train_loss_pred_grad_all = []
        test_loss_pred_grad_all = []

        L1Norm_corr_grad_all = []
        train_loss_corr_grad_all = []
        test_loss_corr_grad_all = []

        # continuation method
        # outer loop for predictor
        # inner loop for corrector
        # predictor = gradient step for loss
        for pareto in tqdm.trange(n_pareto_grad):

            L1Norm_pred = np.zeros(( n_predictor_loss+1,))
            train_loss_pred = np.zeros(( n_predictor_loss+1,))
            test_loss_pred = np.zeros(( n_predictor_loss+1,))
                                        
            # perform a number of gradient steps for prediction
            for pred in range( n_predictor_loss):
                model.train()
                # model predction
                y_pred = model(X_train)
                # compute loss
                Loss = loss(y_pred, y_train)
                # store values for potential pareto point
                L1Norm_pred[pred] = computeL1Norm(model)/weight_length
                train_loss_pred[pred] = Loss.item()

                # compute gradient
                optimizer_adam.zero_grad()
                Loss.backward()
                # perform gradient step
                optimizer_adam.step()

            # model prediction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)
            # store values for potential pareto point
            L1Norm_pred[n_predictor_loss] = computeL1Norm(model)/weight_length
            train_loss_pred[n_predictor_loss] = Loss.item()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                test_loss_pred[n_predictor_loss] = loss(y_pred, y_test).item()

            L1Norm_pred_grad_all.append(L1Norm_pred.copy())
            train_loss_pred_grad_all.append(train_loss_pred.copy())
            test_loss_pred_grad_all.append(test_loss_pred.copy())

            L1Norm_corr = np.zeros((n_corrector_loss+1,))
            train_loss_corr = np.zeros((n_corrector_loss+1,))
            test_loss_corr = np.zeros((n_corrector_loss+1,))

            # inner loop for correction
            for corr in tqdm.trange(n_corrector_loss):
                model.train()
                optimizer.acceleration_step()
                
                # model predction
                y_pred = model(X_train)
                # compute loss
                Loss = loss(y_pred, y_train)
                        
                # store values for potential pareto point
                L1Norm_corr[corr] = computeL1Norm(model)/weight_length
                train_loss_corr[corr] = Loss.item()

                # compute gradient
                optimizer.zero_grad()
                Loss.backward()

                # preform moo proximal gradient step
                optimizer.MOOproximalgradientstep()

            # model predction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)

            # store values for potential pareto point
            L1Norm_corr[n_corrector_loss] = computeL1Norm(model)/weight_length
            train_loss_corr[n_corrector_loss] = Loss.item()
            # Compute training accuracy
            train_acc = get_accuracy(model, X_train, y_train)
            train_acc_all.append(train_acc)
           
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                test_loss_corr[n_corrector_loss] = loss(y_pred, y_test).item()
                # Compute testing accuracy
                test_acc = get_accuracy(model, X_test, y_test)
                test_acc_all.append(test_acc)

            L1Norm_corr_grad_all.append(L1Norm_corr.copy())
            train_loss_corr_grad_all.append(train_loss_corr.copy())
            test_loss_corr_grad_all.append(test_loss_corr.copy())

        test_loss_corr_grad_all = test_loss_corr_grad_all[::-1]
        test_acc_all = test_acc_all[::-1]
        train_acc_all = train_acc_all[::-1]
        L1Norm_corr_grad_all = L1Norm_corr_grad_all[::-1]
        train_loss_corr_grad_all =train_loss_corr_grad_all[::-1]

        return L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, train_acc_all, test_loss_corr_grad_all, test_acc_all    
    
    elif type == 'l1norm_iter':

        L1Norm_pred_prox_all = []
        train_loss_pred_prox_all = []
        test_loss_pred_prox_all = []

        L1Norm_corr_prox_all = []
        train_loss_corr_prox_all = []
        test_loss_corr_prox_all = []

        # continuation method
        # outer loop for predictor
        # inner loop for corrector
        # predictor = gradient step for loss
        for pareto in tqdm.trange(n_pareto_prox):

            L1Norm_pred = np.zeros((n_predictor+1,))
            train_loss_pred = np.zeros((n_predictor+1,))
            test_loss_pred = np.zeros((n_predictor+1,))

            # perform a number of gradient steps for prediction
            for pred in range(n_predictor):
                model.train()
                # model predction
                y_pred = model(X_train)
                # compute loss
                Loss = loss(y_pred, y_train)

                # store values for potential pareto point
                L1Norm_pred[pred] = computeL1Norm(model)/weight_length
                train_loss_pred[pred] = Loss.item()

                # compute gradient
                optimizer.zero_grad()
                Loss.backward()
                # perform gradient step
                optimizer.shrinkage()

            # model predction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)
            # store values for potential pareto point
            L1Norm_pred[n_predictor] = computeL1Norm(model)/weight_length
            train_loss_pred[n_predictor] = Loss.item()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                test_loss_pred[n_predictor] = loss(y_pred, y_test).item()

            L1Norm_pred_prox_all.append(L1Norm_pred.copy())
            train_loss_pred_prox_all.append(train_loss_pred.copy())
            test_loss_pred_prox_all.append(test_loss_pred.copy())

            L1Norm_corr = np.zeros(( n_corr_l1 +1,))
            train_loss_corr = np.zeros(( n_corr_l1 +1,))
            test_loss_corr = np.zeros(( n_corr_l1 +1,))

            # inner loop for correction
            for corr in tqdm.trange( n_corr_l1 ):

                optimizer.acceleration_step()

                # model predction
                y_pred = model(X_train)
                # compute loss
                Loss = loss(y_pred, y_train)
                        
                # store values for potential pareto point
                L1Norm_corr[corr] = computeL1Norm(model)/weight_length
                train_loss_corr[corr] = Loss.item()

                # compute gradient
                optimizer.zero_grad()
                Loss.backward()

                # preform moo proximal gradient step
                optimizer.MOOproximalgradientstep()

            # model predction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)

            # store values for potential pareto point
            L1Norm_corr[ n_corr_l1 ] = computeL1Norm(model)/weight_length
            train_loss_corr[ n_corr_l1 ] = Loss.item()
            train_acc = get_accuracy(model, X_train,y_train)
            train_acc_all.append(train_acc)

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                test_loss_corr[ n_corr_l1 ] = loss(y_pred, y_test).item()
                test_acc = get_accuracy(model, X_train,y_train)
                test_acc_all.append(test_acc)

            L1Norm_corr_prox_all.append(L1Norm_corr.copy())
            train_loss_corr_prox_all.append(train_loss_corr.copy())
            test_loss_corr_prox_all.append(train_loss_corr.copy())

        return L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, train_acc_all, test_loss_corr_prox_all, test_acc_all 


    else:
        raise ValueError(f"Invalid type value: {type}. Must be one of 'first_iter', 'loss_iter', or 'l1norm_iter'.")


#****************For stochastic setting ************************#
def train_test_data(type:str, X_train, y_train,X_test, y_test, model, optimizer, loss, n_corr_first = 0 ,n_pareto_grad = 0, 
                     n_predictor_loss = 0, n_corrector_loss = 0, n_pareto_prox = 0, n_predictor = 0, n_corr_l1 = 0, batch_num = 0):

    # ********** Best loss******

    #best_accuracy = 0.0
    best_loss = float('inf')
    weight_length = n_pareto_prox + n_pareto_grad + 1

    #************ Predictor Optimizer **********
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001) 

    # ********** Accuracy**********
    train_acc_all = []
    test_acc_all = []
   
    #***************************************************

    if type == 'first_iter':

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

                optimizer.acceleration_step()

                # model predction
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
        train_acc = get_accuracy(model, X_train, y_train)
        train_acc_all.append(train_acc)


        #### Testing ***************
        model.eval()
        test_loss_start = np.zeros((n_corr_first+1,))
         
        with torch.no_grad():
            y_pred = model(X_test)
            test_loss_start[n_corr_first] = loss(y_pred, y_test).item()
            # Compute testing accuracy
            test_acc = get_accuracy(model, X_test, y_test)
            test_acc_all.append(test_acc)
            #print('Test Accuracy: %.3f' % test_acc)     

        return  L1Norm_start, train_loss_start,test_loss_start, train_acc_all, test_acc_all 


    elif type == 'loss_iter':

        L1Norm_pred_grad_all = []
        train_loss_pred_grad_all = []

        L1Norm_corr_grad_all = []
        train_loss_corr_grad_all = []
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

                    # optimizer.acceleration_step()

                    # model predction
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

    
            # model prediction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)
            # store values for potential pareto point
            L1Norm_pred[n_predictor_loss] = computeL1Norm(model)/weight_length
            train_loss_pred[n_predictor_loss] = Loss.item()

            L1Norm_pred_grad_all.append(L1Norm_pred.copy())
            train_loss_pred_grad_all.append(train_loss_pred.copy())


            #*** Corrector step for Loss*************#
            L1Norm_corr = np.zeros((n_corrector_loss+1,))
            train_loss_corr = np.zeros((n_corrector_loss+1,))
            test_loss_corr = np.zeros((n_corrector_loss+1,))
           
            # inner loop for correction
            for corr in tqdm.trange(n_corrector_loss):
                model.train()
                # Shuffle the training data
                permutation = torch.randperm(X_train.shape[0])


                # Split the training data into mini-batches
                for i in range(0, X_train.shape[0], batch_num):
                    indices = permutation[i:i+batch_num]
                    batch_X, batch_y = X_train[indices], y_train[indices]

                    optimizer.acceleration_step()
                    
                    # model predction
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


            # model predction
            y_pred = model(X_train)
            # compute loss
            Loss = loss(y_pred, y_train)

            # store values for main pareto point
            L1Norm_corr[n_corrector_loss] = computeL1Norm(model)/weight_length
            train_loss_corr[n_corrector_loss] = Loss.item()

            # Compute training accuracy
            train_acc = get_accuracy(model, X_train, y_train)
            train_acc_all.append(train_acc)                
            #print('corr %d | Train Loss: %.3f, Train Acc: %.3f | Val Loss: %.3f, Val Acc: %.3f' % (corr+1, train_loss_corr[corr], train_acc, val_loss_temp, val_acc))

            L1Norm_corr_grad_all.append(L1Norm_corr.copy())
            train_loss_corr_grad_all.append(train_loss_corr.copy())


            #### Testing**********
            model.eval()
        
            with torch.no_grad():
            # Compute testing accuracy
                y_pred = model(X_test)
                test_loss_corr[n_corrector_loss] = loss(y_pred, y_test).item()
                test_acc = get_accuracy(model, X_test, y_test)
                test_acc_all.append(test_acc)
                #print('Test Accuracy: %.3f' % test_acc)

            test_loss_corr_grad_all.append(test_loss_corr.copy())

        
        test_loss_corr_grad_all = test_loss_corr_grad_all[::-1]
        test_acc_all = test_acc_all[::-1]
        train_acc_all = train_acc_all[::-1]
        L1Norm_corr_grad_all = L1Norm_corr_grad_all[::-1]
        train_loss_corr_grad_all =train_loss_corr_grad_all[::-1]

        return L1Norm_pred_grad_all, train_loss_pred_grad_all, L1Norm_corr_grad_all, train_loss_corr_grad_all, train_acc_all, test_loss_corr_grad_all, test_acc_all
    
    

    elif type == 'l1norm_iter':

        L1Norm_pred_prox_all = []
        train_loss_pred_prox_all = []
       

        L1Norm_corr_prox_all = []
        train_loss_corr_prox_all = []
        test_loss_corr_prox_all = []

        # continuation method
        # outer loop for predictor
        # inner loop for corrector
        # predictor = shrinkage step for L1-Norm
        for pareto in tqdm.trange(n_pareto_prox):

            L1Norm_pred = np.zeros((n_predictor+1,))
            train_loss_pred = np.zeros((n_predictor+1,))
            #val_loss_pred = np.zeros((n_predictor+1,))

            # perform a number of gradient steps for prediction
            
            for pred in range(n_predictor):

                #optimizer.acceleration_step()

                # model predction
                y_pred = model(X_train)
                # compute loss
                Loss = loss(y_pred, y_train)        
                # store values for potential pareto point
                L1Norm_pred[pred] = computeL1Norm(model)/weight_length
                train_loss_pred[pred] = Loss.item()
              # compute gradient
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

                    optimizer.acceleration_step()
                    # model predction
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
            train_acc = get_accuracy(model, X_train, y_train)
            train_acc_all.append(train_acc)
            # Print epoch statistics
            #print('corr %d | Train Loss: %.3f, Train Acc: %.3f | Val Loss: %.3f, Val Acc: %.3f' % (corr+1, train_loss_corr[corr], train_acc, val_loss_temp, val_acc))     

            L1Norm_corr_prox_all.append(L1Norm_corr.copy())
            train_loss_corr_prox_all.append(train_loss_corr.copy())
     
            ##### Testing *************
            test_loss_corr = np.zeros((n_corr_l1+1,))
            model.eval()
            with torch.no_grad():
                # Compute testing accuracy
                y_pred = model(X_test)
                test_loss_corr[n_corr_l1] = loss(y_pred, y_test).item()
                test_acc = get_accuracy(model, X_test, y_test)
                test_acc_all.append(test_acc)
                #print('Test Accuracy: %.3f' % test_acc)

            test_loss_corr_prox_all.append(test_loss_corr.copy())

        return L1Norm_pred_prox_all, train_loss_pred_prox_all, L1Norm_corr_prox_all, train_loss_corr_prox_all, train_acc_all, test_loss_corr_prox_all, test_acc_all 


    else:
        raise ValueError(f"Invalid type value: {type}. Must be one of 'first_iter', 'loss_iter', or 'l1norm_iter'.") 
            
        
        

