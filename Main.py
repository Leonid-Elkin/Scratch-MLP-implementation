import numpy as np
from MLP_class import MLP
from Plotter import plotMovingAverage, plotPolynomialWithTwo
from matplotlib import pyplot as plt

def convertIntoFormat(labels, size): # One hot encode values
    format = np.zeros((len(labels), size)) # Create list of zeros of same length as numbe of classes
    for i, label in enumerate(labels): 
        format[i][label] = 1    # insert a one where nessesary
    return format


def checkAccuracy(lst1, lst2): # gets the acuracy of whether two items with the same indexes in two lists are the same
    result = []
    for a,b in zip(lst1,lst2):
        if a == b:
            result.append(1)
        else:
            result.append(0)
    return np.mean(np.array(result)) # takes mean of list of ones and zeros to get accuracy 


def loadFashionMNIST():
    from tensorflow.keras.datasets import fashion_mnist # import dataset
    (trainingImages, trainingLabels), (testingImages, testingLabels) = fashion_mnist.load_data() # load dataset into lists

    trainingImages = trainingImages.reshape(-1, 28 * 28) / 255.0
    testingImages = testingImages.reshape(-1, 28 * 28) / 255.0 # flatten image into list and divide each pixel value by 255 to make them between one and zero
    trainingLabels = convertIntoFormat(trainingLabels, 10)
    testingLabels = convertIntoFormat(testingLabels, 10) # one-hot-encode all labels

    return trainingImages, testingImages, trainingLabels, testingLabels

def loadMNIST():
    from tensorflow.keras.datasets import mnist

    (trainingImages, trainingLabels), (testingImages, testingLabels) = mnist.load_data()

    trainingImages = trainingImages.reshape(-1, 28 * 28) / 255.0
    testingImages = testingImages.reshape(-1, 28 * 28) / 255.0
    trainingLabels = convertIntoFormat(trainingLabels, 10)
    testingLabels = convertIntoFormat(testingLabels, 10)

    return trainingImages, testingImages, trainingLabels, testingLabels


def showImg(flatImage, label):
    image = flatImage.reshape(28, 28)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f"label: {label}")

    
    plt.show()

def hyperparameterChanges(variable, start = 0, end = 0, step = 0, start2 = None, end2 = None, step2 = None):

    resultList = []
    if variable == 'lr':
        learningR = start + step
        epochs = step2
        while learningR < end:
            if start2 == 'sigmoid':
                mlp = MLP(LAYERS,'sigmoid',maxAllowableOutput,'xavier')
            elif start2 == 'relu':
                mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he')
            mlp.train(trainingImages,trainingLabels,epochs,learningR,batchSize,testingImages,testingLabels)
            results = mlp.predict(testingImages)
            resultList.append(checkAccuracy(results,answers))
            learningR += step
        print(resultList)
        plotMovingAverage([resultList],1,step,['learning rate vs accuracy'])




    elif variable == 'activation':
        activationFunctions = ['sigmoid','relu','leakyrelu'] # list of activation functions to investigate
        layers = [[784,600,10],[784,64,10],[784,32,32,10]] # list of layers to investigate
        accuracyList = []

        for layer in layers:
            
            resultList = []
            for function in activationFunctions:
                if function == 'sigmoid':
                    epochs = start
                    mlp = MLP(layer,function,maxAllowableOutput,'None')
                    loss = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,avgLossToggle=True)
                    resultList.append(loss) # for every network layout create and train an mlp
                else:
                    epochs = start
                    mlp = MLP(layer,function,maxAllowableOutput,'None')
                    loss = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,avgLossToggle=True)
                    resultList.append(loss)
                
                results = mlp.predict(testingImages)
                accuracy = checkAccuracy(results,answers)
                accuracyList.append(accuracy) # generates accuracies for each network
            print(resultList)
            plotMovingAverage(resultList,2,batchSize,activationFunctions) # plots
        print(accuracyList) # prints out accuracies

    elif variable == 'leakycomparison':
        epochs = start2
        while start <= end:
            mlp = MLP(LAYERS,'leakyrelu',maxAllowableOutput,'he',a=start)
            mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels)

            start += step

            results = mlp.predict(testingImages)
            accuracy = checkAccuracy(results,answers)
            resultList.append(accuracy)
        print(resultList)
        plotMovingAverage([resultList],1,step,['a value versus accuracy'])

    elif variable == 'optimizers':
            
            if end == 'sigmoid':
                opt = 'xavier'
            elif end == 'relu' or end == 'leakyrelu':
                opt = 'he'
    
            epochs = start
            mlp = MLP(LAYERS,end,maxAllowableOutput,opt,lossFunc='mean')
            loss = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,avgLossToggle=True)
            resultList.append(loss)
            mlp = MLP(LAYERS,end,maxAllowableOutput,'None',lossFunc='mean')
            loss = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,avgLossToggle=True)
            resultList.append(loss)
            print(resultList)
            plotMovingAverage(resultList,2,1,['xavier optimization function','no optimization function'])


    elif variable == 'epochs':
        epochs = end
        mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he',lossFunc='cross')
        losslist,trainingLossList  = mlp.train(trainingImages,trainingLabels,int(epochs),lr,batchSize,testingImages,testingLabels,True,0.9,True,False,True)
        plotMovingAverage([losslist,trainingLossList],2,labels=['loss recorded on training dataset','loss recorded on testing dataset'])

    elif variable == 'lossFuncs':
        epochs = end
        funcs = ['mean','cross']
        for func in funcs:
            mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he',lossFunc=func,seed = 100)
            mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels)

            results = mlp.predict(testingImages)
            accuracy = checkAccuracy(results,answers)
            resultList.append(accuracy)
            
        print(resultList)

    elif variable == 'EfficiencyFrontier':

        width = start
        epochs = 100
        flopResults = []
        for layerN in range (start2,end2,step2):
            while width < end:
                layers = [784]
                for i in range (layerN):
                    layers.append(width)
                layers.append(10)
                mlp = MLP(layers,'relu',maxAllowableOutput,'he',lossFunc='cross')
                flopList = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,True,0.95,True,True,False,True)
                print(flopList)
                flopResults.append(flopList)
                width += step
        print(flopResults)
        plotPolynomialWithTwo(flopResults,True)

    elif variable == 'LayerSize':
        epochs = start2
        for layerSize in range (start,end,step):
            LAYERS[1] = layerSize

            mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he',lossFunc='cross',seed=100)
            mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,lrDecay=True,decayRate=0.9)
            results = mlp.predict(testingImages)
            accuracy = checkAccuracy(results,answers)
            resultList.append(accuracy)
        
        plotMovingAverage([resultList],1,multiplier=step,labels=["Layer size vs accuracy given"],start=start)

    return None


def loadCIFAR():

    #Change number of input nodes to 3072

    from tensorflow.keras.datasets import cifar10

    (trainingImages, trainingLabels), (testingImages, testingLabels) = cifar10.load_data()

    trainingImages = trainingImages.reshape(-1, 32 * 32 * 3) / 255.0
    testingImages = testingImages.reshape(-1, 32 * 32 * 3) / 255.0
    trainingLabels = convertIntoFormat(trainingLabels, 10)
    testingLabels = convertIntoFormat(testingLabels, 10)

    return trainingImages, testingImages, trainingLabels, testingLabels
 

def trainBestMNIST():
    
    #Best accuracy I got is 0.9851

    #LAYERS = [784,925,10]       
    #maxAllowableOutput = 9e9
    #epochs = 15
    #lr = 0.01
    #batchSize = 50
    #seed = 100
    #cross-entropy loss


    LAYERS = [784,925,10]       
    maxAllowableOutput = 9e9
    epochs = 15
    lr = 0.01
    batchSize = 50
    seed = 100

    mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he',seed,'cross')
    mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,lrDecay=True,decayRate=0.9)
    return mlp

def trainBestFashionMNIST():
    #Best accuracy I got is 0.8967

    LAYERS = [784,128,10]       
    maxAllowableOutput = 500
    epochs = 30
    lr = 0.01
    batchSize = 50
    seed = 104

    mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he',seed,'mean')
    mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,True,0.9,False,False,False,True)
    return mlp

def trainBestCIFAR():
    layers = [3072,1024,10]
    maxAllowableOutput = 500
    epochs = 1
    lr = 0.01
    batchSize = 64
    seed = 100

    mlp = MLP(layers,'sigmoid',maxAllowableOutput,'xavier',seed)
    mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,True,0.9,False,False,False)

    return mlp



LAYERS = [784,64,10]      
lr = 0.01
batchSize = 50
maxAllowableOutput = 500


trainingImages, testingImages, trainingLabels, testingLabels = loadMNIST()

answers = np.argmax(testingLabels, axis=1)
mlp = trainBestMNIST()
results = mlp.predict(testingImages)
accuracy = checkAccuracy(answers,results)
print(f'Final accuracy from the test dataset: {accuracy:.4f} with {LAYERS} node layout')
