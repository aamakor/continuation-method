# Predictor-Corrector Algorithm
## MNIST (Contains both the predictor-corrector and weighted sum algorithms for the mnist and iris data.)
### This file consists of: 
* The predictor-corrector algorithm for the deterministic and stochastic approach on MNIST and iris dataset (plots)*
*T he Weighted-Sum Algorithm for the deterministic and stochastic approach on MNIST dataset (plots)*
* The comparison of both algorithms (plots)*

#### continuationTest.py - Contains all the predictor-corrector algorithm for both the MNIST and Iris dataset.
#### confirguration.yaml- Contains the setting for the predictor-corrector algorithm.
#### DataLoader.py - Contains functions for processing and loading the datasets.
#### helperFunctions.py - Contains functions for modifying our predictor-corrector algorithm if needed.
#### job.py- Contains code for plotting the comparison between weighted sum and predictor-corrector algorithm.
#### OwnDescent.py - Contains our optimizer code.
#### plotResults.py - For any visualistions.
#### weightedsumTest.py- Contains our weighted sum algorithm.

### To install

* pip install continuation_method

### To plot

* Already existing plot: from MNIST import plotResults
* Comparison between predictor-corrector algorithm and weighted sum: from MNIST import job
