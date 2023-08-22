# Predictor-Corrector Algorithm
## predictor-corrector folder: (Contains both the predictor-corrector and weighted sum algorithms for the mnist and iris datasets.)
### This file contains the following: 
* The predictor-corrector algorithm for the deterministic and stochastic approach on Iris and MNIST dataset respectively (plots).
* The Weighted-Sum Algorithm for the stochastic approach on MNIST dataset (plots).
* The comparison of both algorithms for the MNIST dataset (plots). 

    Note :
    * While experimenting on our algorithm using the Iris dataset on deterministic setting, some specific configurations were made such as starting very close to zero.
    * The main file is job.py
    * Ensure you are in the right "directory" before compilation.


#### 1. configuration_iris.yaml
     Contains the setting for the predictor-corrector algorithm Iris dataset.
#### 2. configuration.yaml 
    Contains the setting for the predictor-corrector algorithm MNIST dataset.
#### 3. continuationTest.py
    Contains the training on MNIST and Iris datasets using the predictor-corrector algorithms.
#### 4. DataLoader.py
    Contains functions for processing and loading the datasets.
#### 5. functions.py
    Contains the algorithm of the predictor-corrector method for the deterministic and  stochastic settings
#### 6. helperFunctions.py
    Contains help functions for modifying our predictor-corrector algorithm when needed.
#### 7. job.py
     This is the main file that compiles all the other files and return the different Pareto front plots (run this file to see results and plots)
#### 8. OwnDescent.py
    Contains our multiobjective proximal gradient optimizer class code.
#### 9. perm.txt
    Shuffled indices for the Iris dataset data loading.
#### 10. plotResults.py
    Contains functions for various visualisations.
#### 11. weightedsumTest.py
    Contains the weighted sum algorithm.

### Results_reference :- 
    This folder consists of results obtain to form a baseline, where we have executed Algorithm 2 using very small step sizes. Interestingly, the Pareto set and front consist of multiple components, which we were only able to find by repeated application of the continuation method with random initial conditions (multi-start).

### Results :-
    This folder contains the outputs/results for:
        1.  deterministic CM on Iris dataset (Results_cm_dt_iris). 
        2.  stochastic CM on MNIST dataset (Results_cm_sto).
        3. stochastic WS on MNIST dataset (Results_ws_sto).

### images :-
    This is the folder where all the images plotted are saved.


### Packages:-
    * Python
    * Torch
    * Numpy
    * Scikit-learn (load data)
    * Matplotlib
    * Keras (load data)


