#!/bin/python3
from keras.models import Sequential
from keras.layers import Dense, Activation, GRU
from keras.layers.convolutional import Conv1D
import numpy as np
from numpy import random
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import math
import h5py
import sys

class SignalCleaner:
    def __init__(self):
        self.sample_size = 128
        self.num_samples = 512
        self.INTERNAL_SIZE = 128
        self.RNN_LAYERS = 1
        self.model = Sequential()
        self.model.add(Conv1D(4, 3, strides=2, padding="causal", input_shape=(self.sample_size,1)))
        self.model.add(Conv1D(8, 3, strides=2, padding="causal"))
        self.model.add(Conv1D(16, 3, strides=2, padding="causal"))
        for i in range(self.RNN_LAYERS):
            self.model.add(GRU(self.INTERNAL_SIZE))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


    def segment_sample(self, x, y):
        x_ = np.empty([np.shape(x)[0] - self.sample_size, self.sample_size, 1])
        y_ = np.empty([np.shape(y)[0] - self.sample_size])
        for k in range(0, np.shape(x)[0] - self.sample_size):
            x_[k] = np.reshape(x[k:k+self.sample_size],(self.sample_size,1))
            y_[k] = y[k]
        return x_, y_
        
    def train_on_signal(self, x, y):
        self.model.train_on_batch(x, y)

    def plot_on_signal(self, x, y):
        plt.plot(np.linspace(0, np.shape(x)[0], num=np.shape(x)[0]), x)
        plt.plot(np.linspace(0, np.shape(y)[0], num=np.shape(y)[0]), y)
        x_, y_ = self.segment_sample(x, y)
        pred = self.model.predict_on_batch(x_)
        plt.plot(np.linspace(0, np.shape(pred)[0], num=np.shape(pred)[0]), pred)
        return self.model.evaluate(x_, y_)

    def test_on_signal(self, x, y):
        #x_, y_ = self.segment_sample(x, y)
        return self.model.evaluate(x, y)

    def gen_signal(self):
        w = random.random((2))*4
        b = random.random((2))*2
        b = b / np.linalg.norm(b)
        n = 1 + random.random()*5
        o = random.random()
        y = np.linspace(0, 10, num=self.num_samples+self.sample_size)
        y = np.vectorize(lambda t: math.sin(w[0]*t)*b[0]+math.cos(w[1]*t)*b[1])(y)+o
        r = (random.rand((self.num_samples+self.sample_size))-.5)*n
        x = y + r
        #print(x)
        #print(y)
        return x, y
        
        
if __name__ == "__main__":
    clean = SignalCleaner()
    if len(sys.argv) > 1:
        clean.model.load_weights(sys.argv[1])
    tests = 20
    tx = np.empty([tests, clean.num_samples+clean.sample_size])
    ty = np.empty([tests, clean.num_samples + clean.sample_size])
    for i in range(0,tests):
        x, y = clean.gen_signal()
        tx[i] = x
        ty[i] = y
    tx = np.reshape(tx,[-1])
    ty = np.reshape(ty,[-1])
    tx_, ty_ = clean.segment_sample(tx,ty)
    
    for t in range(0, 10000):
        x, y = clean.gen_signal()
        x_, y_ = clean.segment_sample(x, y) 
        clean.train_on_signal(x_, y_)
        if t % 10 == 0:
            print("epoch:",t)
        #if t % 50 == 0:
            #x, y = clean.gen_signal()
         #   print(clean.test_on_signal(tx_, ty_))

        if t % 100 == 0:
            x, y = clean.gen_signal()
            print(clean.plot_on_signal(tx, ty))
            plt.savefig(str(t)+".png",dpi=1000)
            plt.cla()

        if t % 1000 == 0:
            clean.model.save_weights(str(t)+".h5")
