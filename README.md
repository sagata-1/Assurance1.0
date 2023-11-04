# Explanation of the different files and my code


## Main files
The main files to look at for today are:
- `XGBoost_and_GA.ipynb`
- `old_and_glpk.ipynb` 
and if you want to see the old code
- `old.ipynb`   
- `optimal_tree_fraud.png`
- `optimal_tree_fraud_1.png`

### XGBoost_and_GA.ipynb
This is the main one, it contains my implementation of the GA with XGBoost, which ended up selecting 35 features, and doesn't work too well (can't find any solution within an hour timeframe).

### old_and_glpk.ipynb
Thus is the same algorithm that was used with Gurobi, where XGBoost was used to select the important features, but applied to GLTK, with appropriate paramter adjustments. This one does find a solution, as there aren't too many solutions, but its the heuristic one, where everything is taken to be "Not-Fraud".


### old.ipynb
This is just the original notebook, where everything is as it was with Gurobi

### optimal_tree_fraud.png
This is the result from the most recent run (which is from old_and_glpk.ipynb, as XGBoost_and_GA.ipynb hasn't found a good result).

### optimal_tree_fraud_1.png
This is the best old result, with the 41 Fraud cases identified.

All the notebooks should have explanation with the code to make them more readable.

## Some instructioNS for GLPK installation
On top of doing a pip install of GLPK, you should also do an install using your device's package manager. I'm not sure if Windows has an equivalent, but Mac uses homebrew (which I believe is now available for windows). I believe windows has winget, and Chocolatey. The command for Homebrew that I use is `brew install glpk`. The appropriate tuning of the hyperparameters for GLTK is what I did in the notebook.
