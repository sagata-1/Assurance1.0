# Explanation of the different files and my code- CURRENT STATUS


## Main files
The main files to look at for today are:
- `bnp_2.ipynb`
- `bnp_1.ipynb` 
- `vision.txt`
- `hyper.txt`
- `best_optimal_tree_fraud.png`
- `optimal_tree_fraud.png`

and if you want to see the old code

- `old.ipynb`   
- `old_and_glpk.ipynb`
- `optimal_tree_fraud_1.png`


### bnp_2.ipynb
This is the main one, it contains my implementation of the GA that select for hyperparameters. Not fully tested yet

### bnp_1.ipynb
This contains the GA implementation for feature selection. Its best output for the day has been 42 fraud cases identified. It also has the code for the pure XGBoost run with the right hyperparameters that produced an output of 48 fraud cases identified.

### vision.txt
Text file that describes the design and vision of the GA for hyperparameter selection that I wrote in `bnp_2.ipynb`

### hyper.txt
Text file that has written down the hyperparameters I used for 48 fraud case identification with pure XGBoost from `bnp_1.ipynb`

### old_and_glpk.ipynb
This is the same algorithm that was used with Gurobi, where XGBoost was used to select the important features, but applied to GLTK, with appropriate paramter adjustments. This one does find a solution, as there aren't too many solutions, but its the heuristic one, where everything is taken to be "Not-Fraud".


### old.ipynb
This is just the original notebook, where everything is as it was with Gurobi

### optimal_tree_fraud.png
This is the result from the most recent run (which is from old_and_glpk.ipynb, as XGBoost_and_GA.ipynb hasn't found a good result).

### best_optimal_tree_fraud.png
This is the best old result, with the 48 Fraud cases identified.

All the notebooks should have explanation with the code to make them more readable.

## Some instructions for GLPK installation
On top of doing a pip install of GLPK, you should also do an install using your device's package manager. I'm not sure if Windows has an equivalent, but Mac uses homebrew (which I believe is now available for windows). I believe windows has winget, and Chocolatey. The command for Homebrew that I use is `brew install glpk`. The appropriate tuning of the hyperparameters for GLTK is what I did in the notebook.
