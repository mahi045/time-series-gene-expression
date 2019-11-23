import pandas as pd
import numpy as np

# load the classification file
gene_class = dict()
for line in open('yeast_classification.txt','r'):
    if line == '\n': break
    l = line.split()
    # print(l)
    gene_class[l[1]] = l[2]


print(gene_class)


# remove unnecessary genes from yeast
# f = open('yeast_actual.txt','w')
# for line in open('yeast.xls','r'):
#     gene = line.split()[0]
#     print(gene)
#     if gene in gene_class.keys():
#         f.write(line)
#
# f.close()

df = pd.read_csv('yeast_actual.txt',sep='\t',header=0)
columns = df.columns.values
print(columns)
actual_genes = df['gene'].tolist()
df_alpha = df.iloc[:,69:]
print(df_alpha.columns.values)
matrix_alpha = df_alpha.values
print(matrix_alpha)

f = open('yeast_elu.txt','w')
for i in range(len(matrix_alpha)):
    series = matrix_alpha[i]
    actual_series = pd.Series(series)
    interpolate_series = actual_series.astype(float).interpolate(method='linear',axis=0)
    print('actual',actual_series.tolist())
    print('interp',interpolate_series.tolist())
    if np.isnan(interpolate_series.tolist()).any():
        continue
    s = actual_genes[i] + ' '+ str(gene_class[actual_genes[i]])
    for val in interpolate_series.tolist():
        s = s + ' '+str(round(val,2))
    s = s + '\n'
    print(s)
    f.write(s)
f.close()

