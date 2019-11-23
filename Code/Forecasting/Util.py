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


