import numpy as np 
from random import uniform
from math import exp, log, pi
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class myRegressor():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.meanY = np.mean(self.Y)
        self.N = self.X.shape[0]
        self.covMat = np.empty((self.N, self.N))
        #init values of theats
        self.ThetaK = np.random.random_sample(self.X.shape[1])
        self.ThetaS = uniform(0,1)
        self.ThetaN = uniform(0,1)

    def calcCovMatrix(self)->None:
        for i in range(self.N):
            for j in range(self.N):
                self.covMat[i][j] = self.ThetaS * exp(- np.sum(self.ThetaK * np.square(self.X[i] - self.X[j])))
                if (i == j):
                    self.covMat[i][j] += self.ThetaN

    def getCovMat(self):
        return self.covMat

    def getThetas(self):
        return (self.ThetaS, self.ThetaK, self.ThetaN)

    def calcKStar(self, inputVector):
        kStar = np.empty((self.N))
        for i in range(self.N):
            kStar[i] = self.ThetaS * exp(- np.sum(self.ThetaK * np.square(self.X[i] - inputVector))) 
        return np.transpose(kStar)

    def predict(self, inputVector):
        kStar = self.calcKStar(inputVector)
        #print('shape of k star: ' , kStar.shape)
        wStar = kStar.dot(np.linalg.inv(self.covMat))
        #print('shape of w star:' , wStar.shape)
        #print('shape of Y:' ,self.Y.shape)
        result = (self.Y.dot(wStar)) + ((1 - np.sum(wStar)) * self.meanY)
        #print('shape of result: ' ,result.shape)
        return  result


class Optimizer():
    def __init__(self, reg: myRegressor, X_test, Y_test, X_train, Y_train):
        self.reg = reg
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_train = X_train
        self.Y_train = Y_train
        self.N = self.reg.N
        self.delta = 0.001
        self.learningRate = 0.1
    
    def calcLoss(self):
        K_d = self.reg.getCovMat()
        loss = np.log(np.linalg.norm(K_d)) + 0.5 * (np.transpose(self.Y_train).dot(np.linalg.inv(K_d)).dot(self.Y_train)) + (self.N/2)*log(2*pi)
        return -loss

    def calcError(self):
        predictions = np.empty(self.X_test.shape[0])
        start = time.time()
        for i in range(self.X_test.shape[0]):
            predictions[i] =  self.reg.predict(self.X_test[i])
            #print(predictions[i])
        print('predictions took: %.2f' % (time.time() - start))
        error = mean_squared_error(predictions, self.Y_test)
        print('mean squared error: %f'  % error)
        return error     

    def calcDiff(self,fX,fXDelta, delta):
        return (fXDelta - fX) / delta

    def calcThetaS(self):
        delta = 0.001
        (ThetaS, ThetaK, ThetaN) = self.reg.getThetas()
        currLoss = self.calcLoss()
        tempTheta = ThetaS
        self.reg.ThetaS += delta
        self.reg.calcCovMatrix()
        newLoss = self.calcLoss()
        print('firstLoss: %f' % currLoss)
        print('secondLoss: %f' % newLoss)
        self.reg.ThetaS = tempTheta - self.calcDiff(currLoss, newLoss, delta) 
        while (newLoss > currLoss):
            currLoss = newLoss
            tempTheta = ThetaS
            self.reg.ThetaS = self.reg.ThetaS - delta
            self.reg.calcCovMatrix()
            newLoss = self.calcLoss()
            print('newLoss: %f' %newLoss)
            self.reg.ThetaS = tempTheta - self.calcDiff(currLoss, newLoss, delta)
        self.reg.ThetaS = tempTheta

    def calcThetaK(self):
        start = time.time()
        ThetaK = self.reg.getThetas()[1]
        diffVetor = np.empty(ThetaK.shape)
        fX = self.calcLoss()
        for i in range(ThetaK.shape[0]):
            print(i)
            tempTheta = self.reg.ThetaK[i]
            self.reg.ThetaK[i] += self.delta
            self.reg.calcCovMatrix()
            fXDelta = self.calcLoss() 
            diffVetor[i] =  self.calcDiff(fX, fXDelta, self.delta)
            self.reg.ThetaK[i] = tempTheta
        print('took: %f' % (time.time() - start))


        self.reg.calcCovMatrix()
        print('new Loss =  %f'  % self.calcLoss())



def loadData():
    I1 = None
    I2 = None 
    Cd = None
    for i in range(9):
        if (I1 is None):
            I1 = np.load(os.path.join('training_data/param%s','I1.npy') % i)
            I2 = np.load(os.path.join('training_data/param%s','I2.npy') % i)
            Cd = np.load(os.path.join('training_data/param%s','Cd.npy') % i)
        else:
            I1 = np.concatenate((I1, np.load(os.path.join('training_data/param%s','I1.npy') % i)))
            I2 = np.concatenate((I2, np.load(os.path.join('training_data/param%s','I2.npy') % i)))
            Cd = np.concatenate((Cd, np.load(os.path.join('training_data/param%s','Cd.npy') % i)))
                
    con = np.concatenate((I1, I2),axis=1)

    return con, Cd

 

def main():
    start = time.time()
    X, Y  = loadData()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    print(X.shape)
    #reg = myRegressor(X_train, Y_train)
    # reg.calcCovMatrix()
    # print('cov matrix calc took: %.2f' % (time.time() - start))
    # optimizer = Optimizer(reg, X_test, Y_test, X_train, Y_train)
    # optimizer.calcThetaK()
    # #predictions = np.empty(X_test.shape[0])
    """
    start = time.time()
    for i in range(X_test.shape[0]):
        predictions[i] =  reg.predict(X_test[i])
        #print(predictions[i])
    print('predictions took: %.2f' % (time.time() - start))
    print('mean squared error: %f'  % mean_squared_error(predictions, Y_test))
    """
if __name__ == "__main__":
    main()