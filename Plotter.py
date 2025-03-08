import numpy as np
import matplotlib.pyplot as plt

def plotMovingAverage(dataList, depth, multiplier=1, labels=None, logScale=False,start=0):
    """Plots moving average of input data
        multiplier defines scale of x axis
        start paramater defines start of x axis.
    """
    plt.figure(figsize=(10, 6))

    for i, data in enumerate(dataList): # for every list in datalist
        movingAvg = np.convolve(data, np.ones(depth) / depth, mode='valid') 
        # calculate moving average using convolve mathematical function and a list with only ones
        plt.plot(np.arange(len(movingAvg)) * multiplier+start, movingAvg, label=labels[i]) 
        # plots points and applies x axis scaling

    if logScale: # log scale toggle
        plt.yscale('log')
        plt.xscale('log')

    plt.xlabel("layer size")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend()
    plt.show()


def plotPolynomialWithTwo(data, logScale = False):
    """
    Plots graph with two axis inputs. Takes in list with sublists with two values each, x and y respectively.
    logScale changes both y and x axis to logarithmic
    """

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
