from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from Util import get_patient_data
import numpy as np

NUM_GENE = 3937
NUM_TIME = 9

gene_names, gene_types, data = get_patient_data(file = '../Datasets/GSE20305_series_matrix.txt')
X = data
Y = gene_types
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=None)

clf = SVC(gamma='scale', C = 3.0)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
print("F1 score: ", metrics.accuracy_score(y_test, y_predict))
print("Accuracy: ", metrics.f1_score(np.asarray(y_test), y_predict, average='macro'))




