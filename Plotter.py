import numpy as np
import matplotlib.pyplot as plt

def plotMovingAverage(dataList, depth, multiplier = 1,logScale = False):
    plt.figure(figsize=(10, 6))
    
    for data in dataList:
        data = np.array(data)

        movingAvg = np.convolve(data, np.ones(depth) / depth, mode='valid')
        
        values = np.arange(len(movingAvg)) * multiplier
        
        plt.plot(values, movingAvg)

    
    if logScale:
        plt.yscale('log')
        plt.xscale('log')
    
    plt.title("Moving Averages for ReLU")
    plt.xlabel("Network Layers (scaled)")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

def plotPolynomialWithTwo(data, logScale = False):
    plt.figure(figsize=(10, 6))
    
    for item in data:

        y, x = zip(*item)
        x = np.array(x)
        y = np.array(y)
        
        plt.plot(x, y)
    

    if logScale:
        plt.yscale('log')
        plt.xscale('log')
    
    plt.xlabel('FLOP/s')
    plt.ylabel('loss')
    plt.title('loss vs compute graph for different neural network')
    plt.grid(True)
    plt.show()