from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt # For general plotting
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mean Squared Error (MSE) loss
def lin_reg_loss(theta, X, y):
    # Linear regression model X * theta
    predictions = X.dot(theta)
    # Residual error (X * theta) - y
    error = predictions - y
    # Loss function is MSE
    loss_f = np.mean(error**2)

    return loss_f

# Need to provide a function handle to the optimizer, which returns the loss objective, e.g. MSE
def func_mse(theta):
    return lin_reg_loss(theta, xValid, yValid)

def generate_data_from_gmm(N, pdf_params):
    # Determine dimensionality from mixture PDF parameters
    n = pdf_params['m'].shape[1]
    # Output samples and labels
    X = np.zeros([N, n])
    labels = np.zeros(N)
    
    # Decide randomly which samples will come from each component
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0) # For intervals of classes

    L = np.array(range(1, len(pdf_params['priors'])+1))
    for l in L:
        # Get randomly sampled indices for this component
        indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
        # No. of samples in this component
        Nl = len(indices)  
        labels[indices] = l * np.ones(Nl) - 1
        if n == 1:
            X[indices, 0] =  norm.rvs(pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)
        else:
            X[indices, :] =  multivariate_normal.rvs(pdf_params['m'][l-1], pdf_params['C'][l-1], Nl)
    
    return X, labels


n = 10
NTrain = 50
NTest = 1000

alpha_range = range(0, 22, 2)
beta_range = np.linspace(pow(10,-4), pow(10,4),20)


def generate_dataset(N, alpha): 
    
    a = np.array(10*[np.random.random(1)])

    x_pdf = {}
    x_pdf['m'] = np.array([np.random.random(n)])
    x_pdf['priors'] = np.array([1])
    x_pdf['C']= np.random.random(size = (n,n))

    # Draw Ntrain iid samples of n-dimensional samples of x from this Gaussian pdf.
    x,_ = generate_data_from_gmm(N, x_pdf)

    alpha = 0.5
    z_pdf = {}
    z_pdf['m'] = np.array([np.zeros(n)])
    z_pdf['priors'] = np.array([1])
    z_pdf['C']= alpha*np.identity(n) 
    # Draw Ntrain iid samples of a n-dimensional random variable z from a 0-mean αI-covariance-matrix Gaussian pdf.
    z, _ = generate_data_from_gmm(N, z_pdf)

    v_pdf = {}
    v_pdf['m'] = np.array([np.zeros(n)])
    v_pdf['priors'] = np.array([1])
    v_pdf['C']= np.identity(n) 
    # Draw Ntrain iid samples of a scalar random variable v from a 0-mean unit-variance Gaussian pdf.
    v, _ = generate_data_from_gmm(N, v_pdf)

    # Calculate Ntrain scalar values of a new random variable as follows y = aT (x+z)+v using the samples of x and v.

    y = np.transpose(a) * (x + z) + v

    return x, y, v



# Max Log Likelihood (MSE) loss
def NLL_loss(theta, X, y):
    # Linear regression model X * theta
    predictions = X.dot(theta)
    # Residual error (X * theta) - y
    error = predictions - y
    # Loss function is MSE
    loss_f = np.mean(error**2)

    return loss_f
splits = 5

xNTrain,yNTrain, vTrain = generate_dataset(NTrain, 0.5)
xNTest, yNTest, vTest = generate_dataset(NTest, 0.5)


        

X = xNTrain
labels =  yNTrain
y =  1


pdf = {}
pdf['m'] = np.array([np.zeros(n)])
pdf['priors'] = np.array([1])


MSE_error = np.ones(len(beta_range))

for index in range(len(beta_range)):

    pdf['C']= beta_range[index]*np.identity(n) 
    w,_ = generate_data_from_gmm(1, pdf)
    w = np.transpose(w)

    y = np.transpose(w) * X + vTrain
    k_fold = KFold(n_splits = splits, shuffle = True)
    k = 0

    error = np.ones(splits)


    for train_indices, valid_indices in k_fold.split(X):
        xTrain, xValid = X[train_indices], X[valid_indices]
        yTrain, yValid = labels[train_indices], labels[valid_indices]

        analytical_preds = yValid.dot(w)
        # Minimize using a default unconstrained minimization optimization algorithm
        mse_model = minimize(func_mse, w, tol=1e-6)
        # res is the optimization result, has an .x property which is the solution array, e.g. theta*
        error[k] = mse_model.fun
    #    ax.scatter(x_T[:, 1], mse_preds, color='red', label="MSE")
        k += 1
    MSE_error[index] = np.mean(error, axis=1)

best_beta = beta_range[np.argmin(MSE_error)]

print("Best Beta is: ", best_beta)

pdf = {}
pdf['m'] = np.array([np.zeros(n)])
pdf['priors'] = np.array([1])
pdf['C']= best_beta*np.identity(n) 
w,_ = generate_data_from_gmm(1, pdf)
w = np.transpose(w)
loss = lin_reg_loss(w, xTrain, yTrain)
print("loss = ", loss)
# y = wT x+w0 +v and v is an additive white Gaussian noise (with zero-mean and unit-variance).
# We are unaware of the presence of the noise term z in the true generative process. We think that
# this process also has linear model parameters close to zero, so we use a 0-mean and βI-covariance matrix
# Gaussian pdf as a prior for the model parameters w (which contain w0).