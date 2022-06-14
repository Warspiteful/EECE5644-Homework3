import matplotlib.pyplot as plt # For general plotting
from scipy.stats import norm, multivariate_normal
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=22)  # fontsize of the figure title

class TwoLayerMLP(nn.Module):
    # Two-layer MLP (not really a perceptron activation function...) network class
    
    def __init__(self, input_dim, hidden_dim, C):
        super(TwoLayerMLP, self).__init__()
        # Fully connected layer WX + b mapping from input_dim (n) -> hidden_layer_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        # Output layer again fully connected mapping from hidden_layer_dim -> outputs_dim (C)
        self.output_fc = nn.Linear(hidden_dim, C)
        # Log Softmax (faster and better than straight Softmax)
        # dim=1 refers to the dimension along which the softmax operation is computed
        # In this case computing probabilities across dim 1, i.e., along classes at output layer
        self.log_softmax = nn.LogSoftmax(dim=1) 
        
    # Don't call this function directly!! 
    # Simply pass input to model and forward(input) returns output, e.g. model(X)
    def forward(self, X):
        # X = [batch_size, input_dim (n)]
        X = self.input_fc(X)
        # Non-linear activation function, e.g. ReLU (default good choice)
        # Could also choose F.softplus(x) for smooth-ReLU, empirically worse than ReLU
        X = F.relu(X)
        # X = [batch_size, hidden_dim]
        # Connect to last layer and output 'logits'
        X = self.output_fc(X)
        # Squash logits to probabilities that sum up to 1
        y = self.log_softmax(X)
        return y


# 100,200,500,1000,2000,5000 samples and a test dataset with 100000
NTrain = [100, 200, 500, 1000, 2000, 5000]
NTest = 100000

gmm_pdf = {}

# Class priors
gmm_pdf['priors'] = np.array([0.25, 0.25, 0.25, 0.25])
num_classes = len(gmm_pdf['priors'])
gmm_pdf['m']  = np.array([[-3/2, -3/2, -3/2], [-1/2, -1/2, -1/2], [1/2, 1/2, 1/2], [1, 1, 1]])
gmm_pdf['C']  = np.array([
                [
                [.2, 0, 0],
                [0, .2, 0],
                [0, 0, .2]],
                [
                [0.3, 0, 0],
                [0, 0.3, 0],
                [0, 0, 0.3]],
                [
                [0.3, 0, 0],
                [0, 0.3, 0],
                [0, 0, 0.3]],
                [
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5]]
                ])



# ERM classification rule (min prob. of error classifier)
def perform_map_classification(X, gmm_params, C):    
    # Conditional likelihoods of each x given each class, shape (C, N)
    class_cond_likelihoods = np.array([multivariate_normal.pdf(X, gmm_params['m'][c], gmm_params['C'][c]) for c in range(C)])

    # Take diag so we have (C, C) shape of priors with prior prob along diagonal
    class_priors = np.diag(gmm_params['priors'])
    # class_priors*likelihood with diagonal matrix creates a matrix of posterior probabilities
    # with each class as a row and N columns for samples, e.g. row 1: [p(y1)p(x1|y1), ..., p(y1)p(xN|y1)]
    class_posteriors = class_priors.dot(class_cond_likelihoods)

    # Conditional risk matrix of size C x N with each class as a row and N columns for samples
    
    return np.argmax(class_posteriors, axis=0)




def model_train(model, data, labels, criterion, optimizer, num_epochs=25):
    # Apparently good practice to set this "flag" too before training
    # Does things like make sure Dropout layers are active, gradients are updated, etc.
    # Probably not a big deal for our toy network, but still worth developing good practice
    model.train()
    # Optimize the neural network
    for epoch in range(num_epochs):
        # These outputs represent the model's predicted probabilities for each class. 
        outputs = model(data)
        # Criterion computes the cross entropy loss between input and target
        loss = criterion(outputs, labels)
        # Set gradient buffers to zero explicitly before backprop
        optimizer.zero_grad()
        # Backward pass to compute the gradients through the network
        loss.backward()
        # GD step update
        optimizer.step()
        
    return model

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

def train_model(X, labels, p, C):

    model = TwoLayerMLP(int(X.shape[1]), int(p), C)
    # Stochastic GD with learning rate and momentum hyperparameters
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # The nn.CrossEntropyLoss() loss function automatically performs a log_softmax() to 
    # the output when validating, on top of calculating the negative log-likelihood using 
    # nn.NLLLoss(), while also being more stable numerically... So don't implement from scratch
    criterion = nn.CrossEntropyLoss()
    trained_model = model_train(model, torch.FloatTensor(X),  torch.LongTensor(labels),criterion,optimizer )
    return trained_model


def model_predict(model, data, labels):
    # Similar idea to model.train(), set a flag to let network know your in "inference" mode
    model.eval()
    # Disabling gradient calculation is useful for inference, only forward pass!!
    with torch.no_grad():
        # Evaluate nn on test data and compare to true labels
        predicted_labels = model(data)
        # Back to numpy
        predicted_labels = predicted_labels.detach().numpy()

        predicted_labels = np.argmax(predicted_labels, 1)
        return (predicted_labels != labels).sum().item() / len(labels) 
fig = plt.figure(figsize=(10, 10))

ax_raw = fig.add_subplot(111, projection='3d')



X, labels = generate_data_from_gmm(NTest, gmm_pdf) 

xTest = X
labelsTest = labels 
n = X.shape[1]
L = np.array(range(num_classes))

# Count up the number of samples per class
N_per_l = np.array([sum(labels == l) for l in L])
print(N_per_l)

ax_raw.scatter(X[labels == 0, 0], X[labels == 0, 1], X[labels == 0, 2], c='r', label="Class 0")
ax_raw.scatter(X[labels == 1, 0], X[labels == 1, 1], X[labels == 1, 2], c='b', label="Class 1")
ax_raw.scatter(X[labels == 2, 0], X[labels == 2, 1], X[labels == 2, 2], c='c', label="Class 2")
ax_raw.scatter(X[labels == 3, 0], X[labels == 3, 1], X[labels == 3, 2], c='g', label="Class 3")

ax_raw.set_xlabel(r"$x_1$")
ax_raw.set_ylabel(r"$x_2$")
ax_raw.set_zlabel(r"$x_3$")
# Set equal axes for 3D plots
ax_raw.set_box_aspect((np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(X[:, 2])))

plt.title("Data and True Class Labels")
plt.legend()
plt.tight_layout()
#plt.show()  


decisions = perform_map_classification(X, gmm_pdf, num_classes)

print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, labels)
print(conf_mat)

correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(NTest - correct_class_samples))

prob_error = 1 - (correct_class_samples / NTest)
print("Estimated Probability of Error: {:.4f}".format(prob_error))

splits = 10




def softmax(x):
    # Numerically stable with large exponentials (log-sum-exp trick if you're curious)
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)

num_nodes = range(4, 92, 2)


def model_node_select(X, labels, C):



    k_fold = KFold(n_splits = splits, shuffle = True)
    Error = np.zeros([len(num_nodes), splits])

    for index, nodes in enumerate(num_nodes):
        k = 0
        model = TwoLayerMLP(X.shape[1], nodes, C)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
        for train_indices, valid_indices in k_fold.split(X):
            xTrain, xValid = X[train_indices], X[valid_indices]
            yTrain, yValid = labels[train_indices], labels[valid_indices]
        
            #Trains Model
            trained_model = model_train(model, torch.FloatTensor(xTrain),  torch.LongTensor(yTrain),criterion,optimizer, 25 )
            
 
            y_pred = np.argmax(trained_model(torch.FloatTensor(xValid)).detach().numpy(), axis=1)

            Error[index, k] = np.mean((y_pred - yValid)**2)

            k += 1
    Error_Mean = np.mean(Error, axis=1)

    min_error = np.min(Error_Mean)
    node = num_nodes[np.argmin(Error_Mean)]

    return node, min_error, Error_Mean








node = np.zeros(len(NTrain))
min_error = np.zeros(len(NTrain))
error = np.zeros([len(NTrain),len(num_nodes)])

X = [NTrain[0],NTrain[1], NTrain[2],NTrain[3], NTrain[4], NTrain[5]]
labels = [NTrain[0],NTrain[1], NTrain[2],NTrain[3], NTrain[4], NTrain[5]]

for index in range(len(NTrain)):
    X[index], labels[index] = generate_data_from_gmm(NTrain[index], gmm_pdf) 

node[0], min_error[0], error[0] = model_node_select(X[0],labels[0], num_classes)
node[1], min_error[1], error[1] = model_node_select(X[1],labels[1], num_classes)
node[2], min_error[2], error[2] = model_node_select(X[2],labels[2], num_classes)
node[3], min_error[3], error[3] = model_node_select(X[3],labels[3], num_classes)
node[4], min_error[4], error[4] = model_node_select(X[4],labels[4], num_classes)
node[5], min_error[5], error[5] = model_node_select(X[5],labels[5], num_classes)

for index in range(len(NTrain)):
    print("Sample Size: ",  NTrain[index], "- Optimal # of Nodes: ", node[index] ," MSE: ", min_error[index])


plt.show()

models = []
for index in range(len(NTrain)):
    models.append(train_model(X[index], labels[index],node[index], num_classes))


perror = np.zeros(len(NTrain))
for index in range(len(NTrain)):
    perror[index] = model_predict(models[index], torch.FloatTensor(xTest), labelsTest)

fig = plt.figure(figsize=(10,10))
plt.semilogx(NTrain, prob_error*np.ones(len(perror)), ":")
plt.semilogx(NTrain, perror, "o-")
plt.xlabel('Number of Training Samples')
plt.ylabel('P(error)')
plt.legend(["MAP Error", "NN Error Estimate"])
plt.grid(True)


fig = plt.figure(figsize=(10,10))
plt.plot(NTrain, node, 'o-')
plt.xlabel('Number of Training Samples')
plt.ylabel('Number of Hidden Nodes')
plt.show()



