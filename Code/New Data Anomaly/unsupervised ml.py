from sklearn.svm import OneClassSVM
from Util import get_patient_data
from sklearn.model_selection import train_test_split


gene_names, gene_types, data = get_patient_data()
X = data
Y = gene_types
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=None)

def one_class_svm():
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_train)
    for i in range(len(y_test)):
        print(y_test[i],y_predict[i])
    pass



one_class_svm()