import pandas as pd
import numpy as np

class Gene:
    def __init__(self,gene_num,patient_num,type):
        self.name = str(gene_num)+ '-' + str(patient_num)
        self.time_points = np.zeros(9)
        self.time_points.fill(np.nan)
        self.type = type
    def print(self):
        s = self.name+' '+self.type
        for val in self.time_points:
            s = s + ' '+ str(round(val,3))
        return s

df = pd.read_excel('patient.XLS')
columns = df.columns.values



# x = Gene('o','1','bad')
# gene_list.append(x)
# print(gene_list[0].time_points)
# x.time_points[1] = 3
# print(gene_list[0].time_points)

f  = open('patient_data.txt','w')
gene_list = []
for g in range(6,len(columns)):
    for row in range(len(df)):
        time = int(df.iloc[row,2][1:])
        gene_name = columns[g]
        value = df.iloc[row,g]
        patient_name = df.iloc[row,1]
        patient_type = df.iloc[row,0]
        # print(time, gene_name, patient_name,patient_type,value)
        if time == 0: # new patient
            x = Gene(gene_name,patient_name,patient_type)
            gene_list.append(x)
            x.time_points[0] = value
        else:
            x.time_points[int(time/3)] = value

for g in gene_list:
    # print(g.name, g.type, g.time_points)
    x = pd.Series(g.time_points)
    g.time_points = np.array(x.astype(float).interpolate(method='linear').tolist())
    if np.isnan(g.time_points).any():
        continue
    print(g.print())
    f.write(g.print()+'\n')

f.close()
