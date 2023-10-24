from scipy.spatial import distance_matrix
import numpy as np
import random

class KNN:
    '''
    k nearest neighboors algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new point
    '''

    def __init__(self, k):
        '''
        INPUT :
        - k : is a natural number bigger than 0 
        '''

        if k <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.k = k
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the coordinates of points
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''
        for i in range(len(X)):
            self.X.append(X[i])
            self.y.append(y[i])
                  
       
    def tests(self):
        tt = np.array([1,7,6])
        sorted = np.sort(tt)

        print(tt)
        print(sorted)


    def predict(self,X_new,p):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the coordinates of new points whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new points
        '''

        y_hat = []

        dst = self.minkowski_dist(X_new, p)
        
        for elm in dst:
            idx = np.argpartition(elm, self.k)[:self.k]
            sum = 0
            for i in idx:
                sum+=self.y[i]
            avg = sum/self.k
            if avg>0.5:
                y_hat.append(1)
            elif avg<0.5:
                y_hat.append(0)
            else:
                y_hat.append(random.randint(0,1))

        
        return np.array(y_hat)
    
    def minkowski_dist(self,X_new,p):
        '''
        INPUT : 
        - X_new : is a MxD numpy array containing the coordinates of points for which the distance to the training set X will be estimated
        - p : parameter of the Minkowski distance
        
        OUTPUT :
        - dst : is an MxN numpy array containing the distance of each point in X_new to X
        '''

        dst = []
        for new_point in X_new:
            buffer = []
            for point in self.X:
                distance = ((abs(new_point[0] - point[0]))**p + (abs(new_point[1] - point[1]))**p)**1/p
                buffer.append(distance)
            dst.append(buffer)

    

        return np.array(dst)