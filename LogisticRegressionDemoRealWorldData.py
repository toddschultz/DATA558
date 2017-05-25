# %% Binary logistic regression with l2 regularization
"""
Demonstration of binary logistic regression with l2 regularization. The example
data set is the spam data set from Elements of Statisitical Learning hosted on
https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets. Comments are provide
in this example to help document the algorithm and data processing steps used
here and a comparison to results from the standard package scikit-learn are
also shown.
"""

# Written by Todd Schultz
# 2017

# %% Imports
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.preprocessing
import myLogisticRegression as lr

# %% Example Part A
# Load and standardize spam data set.

# Load spam data set
spam = pd.read_csv('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data', sep=' ', header=None)
testindicator = pd.read_table('https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest', sep=' ', header=None)

# Convert data to numeric matrix format
x = np.asarray(spam)[:, 0:-1]
y = np.asarray(spam)[:, -1]*2 - 1  # Convert to +/- 1
testindicator = np.array(testindicator).T[0]

# Divide the data into train, test sets
xtrain = x[testindicator == 0, :]
xtest = x[testindicator == 1, :]
ytrain = y[testindicator == 0]
ytest = y[testindicator == 1]

# Standardize the data by removing mean and setting standard deviation to 1
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# Keep track of the number of samples and dimension of each sample
ntrain = len(ytrain)
ntest = len(ytest)
p = np.size(x, 1)


# %% Example Part B
# Execute logistic regression model on the spam data set from above.

# Parameters
lambduh = 0.006

# Run algorithm
betafast = lr.fastgraddescend(lr.flogistic, lr.gradflogistic, lambduh, x=xtrain, y=ytrain, maxiter=1000)
lr.objectivePlot(lr.flogistic, betafast, lambduh, x=xtrain, y=ytrain)

# Run scikit-learn logisitic regression
sklr = sklearn.linear_model.LogisticRegression(penalty='l2', C=1/(2*lambduh*ntrain), fit_intercept=False, tol=10e-8, max_iter=1000)
sklr.fit(xtrain, ytrain)

# Compare final objective values and beta valus
print('Final beta coefficients')
print('scikit learn')
print(sklr.coef_)
print('fastgraddescent')
print(betafast[-1, :])

print()
print('Final objective values')
print('scikit learn')
print(lr.flogistic(np.squeeze(sklr.coef_), lambduh, x=xtrain, y=ytrain))
print('fastgraddescent')
print(lr.flogistic(betafast[-1, :], lambduh, x=xtrain, y=ytrain))

# Plot misclassification error
lr.plotMisclassificationError(betafast, x=xtrain, y=ytrain)

# %% Find optimal lambda value
# The value of the regularization parameter, lambda, was found using the
# following commands.

# sklr_cv = sklearn.linear_model.LogisticRegressionCV(penalty='l2', fit_intercept=False, tol=10e-8, max_iter=1000)
# sklr_cv.fit(xtrain, ytrain)
# optimal_lambda = 1/(sklr_cv.C_[0]*2*ntrain)
# print('Optimal lambda=', optimal_lambda)
