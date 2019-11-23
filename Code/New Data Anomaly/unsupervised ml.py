from Util import get_patient_data
import scipy.optimize as opt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    predictions = sigmoid(X @ theta)
    predictions[predictions == 1] = 0.999 # log(1)=0 causes error in division
    error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions);
    return sum(error) / len(y);

def cost_gradient(theta, X, y):
    predictions = sigmoid(X @ theta);
    return X.transpose() @ (predictions - y) / len(y)

gene_names, gene_types, data = get_patient_data(file = '../Datasets/GSE20305_series_matrix.txt')
X = data
Y = gene_types
Y = np.array(Y, dtype = np.intc)
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=None)


numLabels = np.unique(gene_types).shape[0]
np.place(y_train, y_train == numLabels, 0)
np.place(y_test, y_test == numLabels, 0)
numExamples = X_train.shape[0]
numFeatures = X_train.shape[1]

X = np.ones(shape=(X_train.shape[0], X_train.shape[1] + 1))
X[:, 1:] = X_train
classifiers = np.zeros(shape=(numLabels, numFeatures + 1))

for c in range(0, numLabels):
    label = (y_train == c).astype(int)
    initial_theta = np.zeros(X.shape[1])
    classifiers[c, :] = opt.fmin_cg(cost, initial_theta, cost_gradient, (X, label), disp=0)

X = np.ones(shape=(X_test.shape[0], X_test.shape[1] + 1))
X[:, 1:] = X_test
classProbabilities = sigmoid(X @ classifiers.transpose())
predictions = classProbabilities.argmax(axis=1)

print("Training accuracy: ", metrics.accuracy_score(y_test, predictions))
print("F1 score: ", metrics.f1_score(y_test, predictions, average='macro'))



