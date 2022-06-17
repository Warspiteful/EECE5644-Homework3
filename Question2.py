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
from sklearn.linear_model import Ridge 

# Mean Squared Error (MSE) loss
def lin_reg_loss(theta, X, y, v):
    # Linear regression model X * theta

    
    predictions =  X[:,1:].dot(theta[1:]) 
    # Residual error (X * theta) - y
    predictions = predictions + v[:]
    error =  y - predictions 
    # Loss function is MSE
    loss_f = 0.5* np.sum(error**2)  + (1/curr_beta)*theta[1:].T.dot(theta[1:])

    return loss_f



# Mean Squared Error (MSE) loss
def actual_loss(theta, X, y, v):
    # Linear regression model X * theta

    
    predictions = theta[1:].T * X[:,1:] + v 
    # Residual error (X * theta) - y
    error =  y - predictions 
    # Loss function is MSE
    loss_f = 0.5* np.sum(error**2)  + (1/curr_beta)*theta[1:].T.dot(theta[1:])

    return loss_f


# Need to provide a function handle to the optimizer, which returns the loss objective, e.g. MSE
def func_mse(theta):
    return lin_reg_loss(theta, xValid, yValid, vValid)

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

alpha_range = np.linspace(pow(10,-3), pow(10,3),7)
beta_range = np.linspace(-7,5,1000)
beta_range = pow(10,beta_range)

a = np.array(10*[np.random.random(1)])

x_pdf = {}
x_pdf['m'] = np.array([np.random.random(n)])
x_pdf['priors'] = np.array([1])
x_pdf['C']= np.random.random(size = (n,n))

def generate_dataset(N, alpha): 
    




    # Draw Ntrain iid samples of n-dimensional samples of x from this Gaussian pdf.
    x,_ = generate_data_from_gmm(N, x_pdf)

    x = np.transpose(x)


    z_pdf = {}
    z_pdf['m'] = np.array([np.zeros(n)])
    z_pdf['priors'] = np.array([1])
    z_pdf['C']= alpha*np.identity(n) 
    # Draw Ntrain iid samples of a n-dimensional random variable z from a 0-mean αI-covariance-matrix Gaussian pdf.
    z, _ = generate_data_from_gmm(N, z_pdf)

    z = np.transpose(z)

    v_pdf = {}
    v_pdf['m'] = np.array([np.zeros(1)])
    v_pdf['priors'] = np.array([1])
    v_pdf['C']= np.identity(1) 
    # Draw Ntrain iid samples of a scalar random variable v from a 0-mean unit-variance Gaussian pdf.
    v, _ = generate_data_from_gmm(N, v_pdf)

    v = np.transpose(v)


    # Calculate Ntrain scalar values of a new random variable as follows y = aT (x+z)+v using the samples of x and v.

    y = a.T.dot((x + z)) + v


    return x.T, y.T, v.T


def sample_beta_dist(alpha, beta):
    return np.random.beta(alpha, beta)

def analytical_solution(X, y, beta):
    # Analytical solution is (X^T*X)^-1 * X^T * y 
    # Gets (theta)NLL
    return np.linalg.inv(((X-np.mean(X)).T.dot(X) + (1/beta)*np.identity(11))).dot((X- np.mean(X)).T).dot((y-np.mean(y)))



splits = 5
curr_beta = 0
alpha_error = np.zeros(len(alpha_range))

optimal_beta = np.zeros(len(alpha_range))

fig = plt.figure(figsize=(10,10))
plt.xlabel('Beta Value')
plt.ylabel('Error')
for i, alpha in enumerate(alpha_range):
    xNTrain,yNTrain, vTrain = generate_dataset(NTrain, alpha)
    xNTest, yNTest, vNTest = generate_dataset(NTest, alpha)

    X = xNTrain
    labels =  yNTrain
    V = vTrain;
    


    pdf = {}
    pdf['m'] = np.array([np.zeros(n+1)])
    pdf['priors'] = np.array([1])


    X = np.column_stack((np.ones(len(X)), X))  
    xNTest = np.column_stack((np.ones(len(xNTest)), xNTest))  



    MSE_error = np.ones(len(beta_range))
    for index in range(len(beta_range)):


        k_fold = KFold(n_splits = splits, shuffle = True)
        k = 0

        error = np.ones(splits)
        curr_beta = beta_range[index]
        

        for train_indices, valid_indices in k_fold.split(X):
            xTrain, xValid = X[train_indices], X[valid_indices]
            yTrain, yValid = labels[train_indices], labels[valid_indices]
            vTrain, vValid = V[train_indices], V[valid_indices]
            w = analytical_solution(xTrain, yTrain, beta_range[index])

            
            analytical_preds = xValid.dot(w) + vValid
            # Minimize using a default unconstrained minimization optimization algorithm
            mse_model = minimize(func_mse, w, tol=1e-6)
            # res is the optimization result, has an .x property which is the solution array, e.g. theta*
            error[k] = mse_model.fun
        #    ax.scatter(x_T[:, 1], mse_preds, color='red', label="MSE")
            k += 1
        MSE_error[index] = np.mean(error)

    plt.loglog(beta_range, MSE_error, 'o-')
    best_beta = beta_range[np.argmin(MSE_error)]

    print("Best Beta is: ", best_beta)
    xValid = xNTest
    yValid = yNTest
    vValid = vNTest

    
    curr_beta = best_beta
    w = analytical_solution(xNTest, yNTest, best_beta)


    mse_model = minimize(func_mse, w, tol=1e-6)
    loss = mse_model.fun
    print("loss = ", loss)
    alpha_error[i] = loss
    optimal_beta[i] = best_beta


plt.legend(alpha_range)

fig = plt.figure(figsize=(10,10))
plt.plot(alpha_range, optimal_beta, 'o-')
plt.xlabel('Alpha Value')
plt.ylabel('Optimal Beta Value')
fig = plt.figure(figsize=(10,10))
plt.semilogx(optimal_beta, alpha_error, 'o')
plt.xlabel('Optimal Beta Value')
plt.ylabel('Error')
plt.legend(alpha_range)
fig = plt.figure(figsize=(10,10))
plt.scatter(alpha_range, alpha_error)
plt.xlabel('Alpha Value')
plt.ylabel('Error')
plt.show()