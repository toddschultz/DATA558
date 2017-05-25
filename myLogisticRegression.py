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
import matplotlib.pyplot as plt
import copy

# %% Binary logistic regression with l2 regularization Python functions


def flogistic(beta, lambduh, x, y):
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


def gradflogistic(beta, lambduh, x, y):
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


def backtracking(f, grad, beta, lambduh, x, y, alpha=0.5, gamma=0.8, maxiter=1000):
    # Back tracking line search algorithm to find the optimal step size
    # Inputs
    # f = function handle for loss function
    # grad = function handle for gradient function
    # beta = vector of logistic regression coefficients
    # lambduh = regularization parameter
    # x = array of predictors (observations by features)
    # y = vector of know classifications
    # alpha = search control parameter
    # gamma = step size control parameter
    # maxiter = maximum allowed iterations
    # Output
    # eta = optimal step size for gradient descent
    
    # Initial step size guess
    eta = 1/(np.linalg.eigvals(1/len(y)*x.T.dot(x)).max() + lambduh)
    
    # Terms for the inequality condition for eta
    fvalue = f(beta, lambduh, x, y)
    gradbeta = grad(beta, lambduh, x, y)
    normgradbeta2 = np.linalg.norm(gradbeta)**2
    
    iter = 0
    while (f(beta - eta*gradbeta, lambduh, x, y) > (fvalue - alpha*eta*normgradbeta2) and iter < maxiter):
        iter = iter + 1
        eta = gamma*eta
    
    if iter == maxiter:
        raise('Maximum number of backtracking iterations reached')
        
    return(eta)


def fastgraddescend(f, grad, lambduh, x, y, maxiter=1000, etatol=1e-12, ftol=1e-12):
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
    
    fvalue = f(beta, lambduh, x, y)
    gradbeta = grad(beta, lambduh, x, y)
    
    # iterate to find minimum
    iter = 0
    eta = 1
    df = 1
    while (iter < maxiter and eta > etatol and df > ftol):
        eta = backtracking(f, grad, beta, lambduh, x, y, alpha=0.5, gamma=0.8, maxiter=1000)
        beta = theta - eta*grad(theta, lambduh, x, y)
        theta = beta + (beta - betam1)*iter/(iter+3)
        betam1 = copy.copy(beta)
        theta_vals = np.vstack((theta_vals, theta))
        iter = iter + 1
        
        # compute change in objective function value
        fvalue = np.vstack((fvalue, f(theta, lambduh, x, y)))
        df = abs(fvalue[-2] - fvalue[-1])

    return(theta_vals)


def myLogisticReg(lambduh, x, y, maxiter=1000):
    # myLogisticReg computes the logistic regression coefficients to the
    # provided training data with an L-2 norm regularization parameter.
    # Inputs
    # lambduh = value of the regularization parameter
    # x = input/predictor training data
    # y = output/response training labels
    betas = fastgraddescend(flogistic, gradflogistic, lambduh, x, y, maxiter=1000)
    return betas


# Misclassication error
def computeMisclassificationError(beta_opt, x, y):
    ypred = 1/(1+np.exp(-x.dot(beta_opt))) > 0.5
    ypred = ypred*2 - 1             # Convert to +/- 1
    miserror = np.mean(ypred != y)  # estimate fraction of data misclassified
    return miserror


# %% Plotting functions
def objectivePlot(f, betas, lambduh, x, y):
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
        objs[i] = f(betas[i, :], lambduh, x, y)
    fig, ax = plt.subplots()
    ax.plot(range(1, npts + 1), objs)
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('Objective value vs. iteration when lambda=' + str(lambduh))


# Plot misclassification error
def plotMisclassificationError(betas, x, y):
    niter = np.size(betas, 0)
    errorMisclass = np.zeros(niter)
    
    for i in range(niter):
        errorMisclass[i] = computeMisclassificationError(betas[i, :], x, y)
        
    fig, ax = plt.subplots()
    ax.plot(range(1, niter + 1), errorMisclass, label='Misclassification Error')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification error')
