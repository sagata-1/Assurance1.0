# Code report-November 26th 2023
## Main Focus and Set-Up
The overall main focus for the past 3-4 days has been to optimize the GA so it gives more accurate predictions. Essentially, the problem was that GA was not behaving ideally no matter what metric was used for fitness evaluation of the trees. If accuracy was used, the the GA was able to identify a 90% accurate decision tree, but in actuality, it was completely useless, since the tree's output was to predict that everything was "Not-Fraud". Alternatively, when using insensitivity and imprecision as the accuracy measures (where imprecision is the proportion of "Not-Fraud" cases evaluated incorrectly and insensitivy the same for "Fraud"), the output was 78% accurate i.e. it got stuff wrong more often than a flat, predict everything as "Not-Fraud" case, which is not ideal.

### Important lines of code (come back to this after you read all the steps, it'll make more sense then)
- All altered code lines are in `large_serial.py`
- Look at lines 57-104 for the XGBoost section
- Look at lines 193-229 for the estimate prediction error function and stuff I changed there
- The most impotant lines in estimate prediction error were 221-226
- Look at line 121 for my current GA population and generations numbers (in that order, you should probably see 30 and 200, 30 for population number, 200 for number of generations. If its running too slow on your computer, just set it to 30, 30, or 20, 20)
- There are a few other lines altered, but they were mostly global variable management, and data size fixes, so I'm not putting those in the README, you can probably see them in the commit history if you really want to.
- As of time of completion of this README (4:58 pm), the line which is in use for fitness evaluation should be line 223, where I'm penalizing inaccuracies + a weight times number of non-fraud and fraud misclassifications (where I've weighted non-fraud misclassifications more).

### Step 1- Add XGBoosting feature selection
The first part of today was basically porting in XGBoost for feature selection. This wasn't anything too complicated, mostly just using the same code used for BNP-OCT, adjusted to affect our current dataset, and make it specific to the feature sizes we were using in this dataset (i.e., there were a couple of global variable bugs where the function thought a variable existed, but it didn't, but that was fixed quickly)

### Step 2: Altering the fitness function
Going back to step 2, I realized mainly that the fitness function that the GA was minimizing needed to be re-evaluated. We already had an accuracy of pure no-fraud of 90%, so if I could then try and force the GA to select trees that put a large weight on accuracy, but also minimized the number of incorrect fraud classifications, we'd have a much better function. After a bit of trial end error, I ended up weighting the penalty at about 0.4 for penalizing the number if incorrect fraud errors (line 225 in `large_serial.py`). But, I realized that the main problem with that is then the "Non-Fraud" cases can be misclassified for similar accuracy, which was a problem. I also couldn't see precisely how many misclassifications of "Non-Fraud" and "Fraud" were occurring, so the next step was to get all the appropriate metrics so I could select the best possible function for GA to minimize.

### Step 3: Adding in appropriate metrics to check
At this stage, I basically made the fitness function return a tuple of 4 things. The first was the function it is minimizing, the second was the normalization factor (this is just the number you subtract at the end to see the total misclassifcation error, rather than the minimizing function i.e. if the GA is minimizing accuracy + something, then the normalizer says, lets just subtract the something, so we know the final accuracy (or, more specifically the inaccuracy) of the tree that we get at the end). The third thing the function returned is the imprecision and the 4th the insensitivity. So, by the end, I could basically see not just the accuracy of the final tree that GA selected for, but also the proportion of Fraud and Non-Fraud misclassifications.

## Step 4: Back to altering the fitness function (and also population and number of generations of GA)
Here, I tried a few different functions to minimize, a mix of first just the fraud cases + inaccuracies, then non-fraud + fraud + inaccuracies, then different weights of these. The important lines for this were lines 221-226, of `large_serial.py`, and you should see the functions either commented out or in use. The error value variable is the result that is being minimized by the GA.

## Step 5:
Tried f1 function, f1 + accuracy + weight precision + weighted accuracy 

## Step 6:
Oversampling tried, depth 4 tree tried, over all combinations of depth 3 so far. Possible next steps include going into GA literature review, poisson distribution or autmating tree depth so that becomes another parameter that GA can choose from.