# Predictor-Corrector Algorithm
## MNIST (Contains both the predictor-corrector and weighted sum algorithms for the mnist and iris data.)
### This file consists of: 
* The predictor-corrector algorithm for the deterministic and stochastic approach on MNIST and iris dataset (plots)*
*T he Weighted-Sum Algorithm for the deterministic and stochastic approach on MNIST dataset (plots)*
* The comparison of both algorithms (plots)*

#### 1. continuationTest.py - Contains all the predictor-corrector algorithm for both the MNIST and Iris dataset.
#### 2. confirguration.yaml- Contains the setting for the predictor-corrector algorithm.
#### 3. DataLoader.py - Contains functions for processing and loading the datasets.
#### 4. helperFunctions.py - Contains functions for modifying our predictor-corrector algorithm if needed.
#### 5. job.py- Contains code for plotting the comparison between weighted sum and predictor-corrector algorithm.
#### 6. OwnDescent.py - Contains our optimizer code.
#### 7. plotResults.py - For any visualistions.
#### 8. weightedsumTest.py- Contains our weighted sum algorithm.

### To install

* pip install -i https://test.pypi.org/simple/ continuation-method

### To plot
* Must be in a Python environment
* Already existing plot: from MNIST import plotResults
* Comparison between predictor-corrector algorithm and weighted sum: from MNIST import job
