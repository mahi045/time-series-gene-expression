from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from Util import get_patient_data


NUM_GENE = 3937
NUM_TIME = 9

gene_names, gene_types, data = get_patient_data()
X = data
Y = gene_types
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=None)
print(X_train.shape)
print(X_test.shape)




