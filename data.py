import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import config

class cifar_data:
    def __init__(self):
        self.trainX,self.trainY = self.train_data()
        self.testX,self.testY = self.test_data()
        self.batch_size = config.batch_size
        self.now_idx = 0
        self.test_idx = 0
    
    def unpickle(self,filename):
        with open(filename,'rb') as f:
            dicts = pickle.load(f,encoding = 'bytes')
        return dicts
    
    def one_hot(self,x,n):
        return (np.arange(n) == x[:,None]).astype(np.integer)
    
    def train_data(self):
        trainX = []
        trainY = []
        for idx in range(1,6):    
            data = self.unpickle(config.path + '\data_batch_' + str(idx))
            X = data[b'data']
            X = np.array(X).reshape(10000,3,32,32).transpose((0,2,3,1))
            label = data[b'labels']
            label = np.array(label)
            label = self.one_hot(label,10)
            if idx == 1:
                trainX = X
                trainY = label
            else:
                trainX = np.concatenate((trainX,X))
                trainY = np.concatenate((trainY,label))
        return trainX,trainY
        
    def test_data(self):
        data = self.unpickle(config.path + '\\test_batch')
        X = data[b'data']
        X = np.array(X).reshape(10000,3,32,32).transpose((0,2,3,1))
        label = data[b'labels']
        label = np.array(label)
        label = self.one_hot(label,10)
        return X,label
    
    def next_batch(self):
        X = []
        Y = []
        count = 0
        while count < self.batch_size:
            if self.now_idx == self.trainX.shape[0]:
                self.now_idx = 0
            if count == 0:
                X = self.trainX[self.now_idx]
                X = X.reshape(1,32,32,3)
                Y = self.trainY[self.now_idx]
            else:
                X = np.vstack((X,self.trainX[self.now_idx].reshape(1,32,32,3)))
                Y = np.vstack((Y,self.trainY[self.now_idx]))
            count = count + 1
            self.now_idx = self.now_idx + 1
        return X / 255.0,Y

    def next_test_batch(self):
        X = []
        Y = []
        count = 0
        while count < self.batch_size:
            if count == 0:
                X = self.testX[self.test_idx]
                X = X.reshape(1,32,32,3)
                Y = self.testY[self.test_idx]
            else:
                X = np.vstack((X,self.testX[self.test_idx].reshape(1,32,32,3)))
                Y = np.vstack((Y,self.testY[self.test_idx]))
            count = count + 1
            self.test_idx = self.test_idx + 1
        return X / 255.0,Y
