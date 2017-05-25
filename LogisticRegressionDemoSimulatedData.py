# %% Binary logistic regression with l2 regularization
"""
Demonstration of binary logistic regression with l2 regularization. The example
data set is a simulated data set using the numpy random number generator.
Comments are provide in this example to help document the algorithm and data
processing steps used here and a comparison to results from the standard
package scikit-learn are also shown.
"""

# Written by Todd Schultz
# 2017

# %% Imports
import numpy as np
import sklearn.linear_model
import sklearn.preprocessing
import myLogisticRegression as lr

# %% Create simulated data
# Load and standardize spam data set.

# Create data for two class that have different means in the data
x1 = 2.1*np.random.randn(1100, 5) + 3
y1 = np.ones(1100)
x2 = 1.3*np.random.randn(1100, 5) + 7.1
y2 = -np.ones(1100)

xtrain = np.vstack((x1[0:1000, :], x2[0:1000, :]))
ytrain = np.hstack((y1[0:1000], y2[0:1000]))
xtest = np.vstack((x1[1000:, :], x2[1000:, :]))
ytest = np.hstack((y1[1000:], y2[1000:]))

# Standardize the data by removing mean and setting standard deviation to 1
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

# Keep track of the number of samples and dimension of each sample
ntrain = len(ytrain)
ntest = len(ytest)
p = np.size(xtrain, 1)


# %% Example Part B
# Execute logistic regression model on the spam data set from above.

# Parameters
lambduh = 0.00069

# Run algorithm
betas = lr.myLogisticReg(lambduh, x=xtrain, y=ytrain, maxiter=1000)
lr.objectivePlot(lr.flogistic, betas, lambduh, x=xtrain, y=ytrain)

# Run scikit-learn logisitic regression
sklr = sklearn.linear_model.LogisticRegression(penalty='l2', C=1/(2*lambduh*ntrain), fit_intercept=False, tol=10e-8, max_iter=1000)
sklr.fit(xtrain, ytrain)

# Compare final objective values and beta valus
print('Final beta coefficients')
print('scikit learn')
print(sklr.coef_)
print('myLogisticRegression')
print(betas[-1, :])

print()
print('Final objective values')
print('scikit learn')
print(lr.flogistic(np.squeeze(sklr.coef_), lambduh, x=xtrain, y=ytrain))
print('myLogisticRegression')
print(lr.flogistic(betas[-1, :], lambduh, x=xtrain, y=ytrain))

# Plot misclassification error
lr.plotMisclassificationError(betas, x=xtrain, y=ytrain)

# %% Find optimal lambda value
# The value of the regularization parameter, lambda, was found using the
# following commands.

# sklr_cv = sklearn.linear_model.LogisticRegressionCV(penalty='l2', fit_intercept=False, tol=10e-8, max_iter=1000)
# sklr_cv.fit(xtrain, ytrain)
# optimal_lambda = 1/(sklr_cv.C_[0]*2*ntrain)
# print('Optimal lambda=', optimal_lambda)
