import numpy as np
from knn import KNN
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import KNeighborsClassifier

N_neig = 18


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
    y_li = y_li[1:]

    k = KNN(N_neig)
    k.train(X_train, y_train)

    print(k.best_k(X_val, y_li, 93))
    # knn_predict = k.predict(X_val,93)
    # acc = np.sum(y_li == knn_predict)/len(knn_predict) * 100
    # print(f'Accuracy: {acc}%')


    # neigh = KNeighborsClassifier(n_neighbors=N_neig)
    # neigh.fit(X_train, y_train)
    # scikit_predict = neigh.predict(X_val)

    
        

if __name__ == '__main__':
    main()

