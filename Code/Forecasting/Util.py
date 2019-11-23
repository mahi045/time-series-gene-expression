def create_dataset():
    f = open('yeast.txt','r')
    f.readline()
    data = []
    # fig = plt.figure()
    # fig.suptitle('all_points')
    for line in f:
        lst = line.split()
        lst = lst[1:]
        lst = [float(i) for i in lst]
        # plt.plot(lst)
        data.append(lst)
    # plt.draw()
    # print('drawn')
    f.close()
    return data
def create_GSE_dataset(file):
    DATASET_DIR = '../Datasets/'

    f = open(DATASET_DIR + file,'r')
    f.readline()
    data = []
    # fig = plt.figure()
    # fig.suptitle('all_points')
    for line in f:
        lst = line.split()
        lst = lst[2:]     ## skip class label and seq id
        lst = [float(i) for i in lst]
        # plt.plot(lst)
        data.append(lst)
    # plt.draw()
    # print('drawn')
    f.close()
    return data


