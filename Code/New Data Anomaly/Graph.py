from Util import *
import matplotlib.pyplot as plt
import random

def patient_data_plot():
    gene_names, gene_types, data = get_patient_data()
    bad_genes = []
    good_genes = []
    k = 5
    for i in range(len(gene_types)):
        if gene_types[i] == 'good':
            good_genes.append(data[i])
        else:
            bad_genes.append(data[i])
    X = list(range(9))
    selected_genes = random.sample(bad_genes,k)
    for gene in selected_genes:
        plt.plot(X,gene)
    plt.show()


def yeast_data_graph():
    gene_names, gene_types, data = get_yeast_data(filename='yeast_alpha.txt')
    selected_genes = []
    k = 5
    type = 'M/G1'
    for i in range(len(gene_types)):
        if gene_types[i] == type:
            selected_genes.append(data[i])

    X = list(range(len(data[0])))
    X_tics = list(range(1,len(data[0]),4))
    k = min(len(selected_genes),k)
    selected_genes = random.sample(selected_genes,k)
    for gene in selected_genes:
        plt.plot(X,gene)
    plt.xticks(X_tics)
    plt.show()

def plot_anomaly_score():
    acc1 = [[72.2	,68.45,	56.4,		41.2],[68.45,74.5,	50.45,		37.4]]
    acc1=np.array(acc1)
    color = ['b','g','r','y']
    methods = ['CNN','LSTM','SVM','One-Class SVM']
    x = np.array([10,18])
    for i in range(4):
        plt.bar(x+1.1*i,acc1[:,i],width=1,label=methods[i])


    plt.ylabel('F1 Score',fontsize=16)
    plt.xticks([12,20],['patinet','yeast'],fontsize=14)
    plt.legend(loc='best',ncol=2)
    plt.xlim(8,25)
    plt.ylim(0,100)
    plt.show()
    pass

plot_anomaly_score()