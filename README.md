# DATA558
Files for DATA 558 Statistical Machine Learning For Data Scientists Code Sharing Homework. 

The Python files included in this repo carry out standard binary ridge logistic regression using a fast gradient
descent algorithm with Nesterov momentum. The computation functions are provided in the file 'myLogisticRegression.py'.
Two Python demonstrations scripts are provided to demonstrate the use of the logistic regression functions. The first
demonstration script, 'LogisticRegressionDemoSimulatedData.py', uses the numpy random number generator to create a
simulated data set for testing. The second demonstration script, 'LogisticRegressionDemoRealWorldData.py', uses the
publicly available Spam data set from https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets, however the script
will automatically download the data. For both demonstration scripts, the results of myLogisticRegression are compared
against a logistic regression model created from scikit-learn. The results from the two methods show good agreement. 

