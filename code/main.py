import numpy as np
from knn import KNN
import csv
from sklearn.neighbors import KNeighborsClassifier

N_neig = 9

def read_data(filename):
    X = []
    y = []

    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        head = True
        for row in datareader:
            if head:
                head = False
                continue
            row = [float(i) for i in row]
            point = np.array(row[:2])
            X.append(point)
            y.append(row[2])
    
    return (X, y)



def main():
    X_li, y_li = read_data("./data/training.csv")
    X_train = np.array(X_li[1:])
    y_train = np.array(y_li[1:])

    X_li, y_li = read_data("./data/validation.csv")
    X_val = np.array(X_li[1:])

    k = KNN(N_neig)
    neigh = KNeighborsClassifier(n_neighbors=N_neig)


    k.train(X_train, y_train)
    neigh.fit(X_train, y_train)

    knn_predict = k.predict(X_val[:19],50)
    scikit_predict = neigh.predict(X_val[:19])

    print(knn_predict)
    print(scikit_predict)

    same_as_scikit = True
    for i in range(len(scikit_predict)):
        if same_as_scikit == False:
            continue
        if knn_predict[i] != scikit_predict[i]:
            same_as_scikit = False
    
    if(same_as_scikit):
        print("Accurate")
    else:
        print("Not accurate")
        

if __name__ == '__main__':
    main()

