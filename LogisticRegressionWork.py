# %% Binary logistic regression with l2 regularization
# Implementation and demonstration of binary logistic regression with l2 
# regularization. The example data set is the spam data set from Elements of 
# Statisitical Learning hosted on 
# https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets
# Comments are provide in this example to help document the algorithm and data
# processing steps used in this demonstration. 

# Written by Todd Schultz
# 2017

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing
import copy

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

# %% Binary logistic regression with l2 regularization Python functions


def flogistic(beta, lambduh, x=xtrain, y=ytrain):
    # Binary logistic regression loss function equation with l2 regularization
    # Inputs
    # beta = vector of logistic regression coefficients
    # lambduh = regularization parameter
    # x = array of predictors (observations by features)
    # y = vector of know classifications
    # Output
    # f = binary logistic regression loss function value
    term1 = np.mean(np.log(1 + np.exp(-y*x.dot(beta))))  # logistic regression
    term2 = lambduh*np.linalg.norm(beta)**2              # ridge regularization
    f = term1 + term2
    return(f)


def gradflogistic(beta, lambduh, x=xtrain, y=ytrain):
    # Gradient of the binary logistic regression loss function equation with
    # l2 regularization
    # Inputs
    # beta = vector of logistic regression coefficients
    # lambduh = regularization parameter
    # x = array of predictors (observations by features)
    # y = vector of know classifications
    # Output
    # gf = vector of gradient binary logistic regression loss function

    # compute gradient terms for the logistic regression part of the loss function     
    expterm = np.exp(-y*x.dot(beta))
    num = y[:, np.newaxis]*x*expterm[:, np.newaxis]
    den = 1 + expterm[:, np.newaxis]
    
    # construct the gradient for the full logistic ridge regression loss function
    term1 = -np.mean(num/den, axis=0)   # logistic regression gradient
    term2 = 2*lambduh*beta              # ridge regularization gradient
    gf = term1 + term2
    
    return(gf)


def backtracking(f, grad, beta, lambduh, alpha=0.5, gamma=0.8, x=xtrain, y=ytrain, maxiter=1000):
    # Back tracking line search algorithm to find the optimal step size
    # Inputs
    # f = function handle for loss function
    # grad = function handle for gradient function
    # beta = vector of logistic regression coefficients
    # lambduh = regularization parameter
    # alpha = search control parameter
    # gamma = step size control parameter
    # x = array of predictors (observations by features)
    # y = vector of know classifications
    # maxiter = maximum allowed iterations
    # Output
    # eta = optimal step size for gradient descent
    
    # Initial step size guess
    eta = 1/(np.linalg.eigvals(1/len(y)*x.T.dot(x)).max() + lambduh)
    
    # Terms for the inequality condition for eta
    fvalue = f(beta, lambduh, x=x, y=y)
    gradbeta = grad(beta, lambduh, x=x, y=y)
    normgradbeta2 = np.linalg.norm(gradbeta)**2
    
    iter = 0
    while (f(beta - eta*gradbeta, lambduh, x=x, y=y) > (fvalue - alpha*eta*normgradbeta2) and iter < maxiter):
        iter = iter + 1
        eta = gamma*eta
    
    if iter == maxiter:
        raise('Maximum number of backtracking iterations reached')
        
    return(eta)


def fastgraddescend(f, grad, lambduh, x=xtrain, y=ytrain, maxiter=1000, etatol=1e-12, ftol=1e-12):
    # Gradient descent minimization algorithm with Nesterov momentum
    # Inputs
    # f = function handle for loss function
    # grad = function handle for gradient function
    # lambduh = regularization parameter
    # x = array of predictors (observations by features)
    # y = vector of know classifications
    # maxiter = maximum allowed iterations
    # etatol = tolerance for minimum step size
    # ftol = tolerance for minimum change in objective function value
    # Output
    # beta_vals = array of regression coefficient for each iteration (iterations by features)
    
    # Initialize variables
    theta = np.zeros(x.shape[1])
    beta = np.zeros(x.shape[1])
    betam1 = copy.copy(beta)
    theta_vals = copy.copy(beta)
    
    fvalue = f(beta, lambduh, x=x, y=y)
    gradbeta = grad(beta, lambduh, x=x ,y=y)
    
    # iterate to find minimum
    iter = 0
    eta = 1
    df = 1
    while (iter < maxiter and eta > etatol and df > ftol):
        eta = backtracking(f, grad, beta, eta, lambduh, x=x, y=y)
        beta = theta - eta*grad(theta, lambduh, x=x, y=y)
        theta = beta + (beta - betam1)*iter/(iter+3)
        betam1 = copy.copy(beta)
        theta_vals = np.vstack((theta_vals, theta))
        iter = iter + 1
        
        # compute change in objective function value
        fvalue = np.vstack((fvalue, f(theta, lambduh, x=x, y=y)))
        df = abs(fvalue[-2] - fvalue[-1])

    return(theta_vals)


def objectivePlot(f, betas, lambduh, x=xtrain, y=ytrain):
    # Plot objective function values over iterations
    # Inputs
    # f = function handle for loss function
    # betas = array of regression coefficient for each iteration
    # lambduh = regularization parameter
    # x = array of predictors (observations by features)
    # y = vector of know classifications
    
    npts = np.size(betas, 0)
    objs = np.zeros(npts)
    for i in range(0, npts):
        objs[i] = f(betas[i, :], lambduh, x=x, y=y)
    fig, ax = plt.subplots()
    ax.plot(range(1, npts + 1), objs)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('Objective value vs. iteration when lambda=' + str(lambduh))
    
    
# Misclassication error
def computeMisclassificationError(beta_opt, x, y):
    y_pred = 1/(1+np.exp(-x.dot(beta_opt))) > 0.5
    y_pred = y_pred*2 - 1       # Convert to +/- 1
    return np.mean(y_pred != y)


# Plot misclassification error
def plotMisclassificationError(betas, x, y, save_file='', title=''):
    niter = np.size(betas, 0)
    errorMisclass = np.zeros(niter)
    
    for i in range(niter):
        errorMisclass[i] = computeMisclassificationError(betas[i, :], x, y)
        
    fig, ax = plt.subplots()
    ax.plot(range(1, niter + 1), errorMisclass, label='Misclassification Error')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification error')
    if title:
        plt.title(title)
    ax.legend(loc='upper right')
    if not save_file:
        plt.show()
    else:
        plt.savefig(save_file)

# %% Example Part B
# Execute logistic regression model on the spam data set from above.

# Parameters
lambduh = 0.006

# Run algorithm
betafast = fastgraddescend(flogistic, gradflogistic, lambduh, x=xtrain, y=ytrain, maxiter=1000)
objectivePlot(flogistic, betafast, lambduh, x=xtrain, y=ytrain)

# Run scikit-learn logisitic regression
lr = sklearn.linear_model.LogisticRegression(penalty='l2', C=1/(2*lambduh*ntrain), fit_intercept=False, tol=10e-8, max_iter=1000)
lr.fit(xtrain, ytrain)

# Compare final objective values and beta valus
print('Final beta coefficients')
print('scikit learn')
print(lr.coef_)
print('fastgraddescent')
print(betafast[-1, :])

print()
print('Final objective values')
print('scikit learn')
print(flogistic(np.squeeze(lr.coef_), lambduh, x=xtrain, y=ytrain))
print('fastgraddescent')
print(flogistic(betafast[-1, :], lambduh, x=xtrain, y=ytrain))

# Plot misclassification error 
plotMisclassificationError(betafast, x=xtrain, y=ytrain)

# %% Find optimal lambda value
# The value of the regularization parameter, lambda, was found using the 
# following commands.

# lr_cv = sklearn.linear_model.LogisticRegressionCV(penalty='l2', fit_intercept=False, tol=10e-8, max_iter=1000)
# lr_cv.fit(xtrain, ytrain)
# optimal_lambda = lr_cv.C_[0]
# print('Optimal lambda=', optimal_lambda)
