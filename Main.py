import numpy as np
from MLP_class import MLP
from Plotter import plotMovingAverage, plotPolynomialWithTwo
from matplotlib import pyplot as plt

def convertIntoFormat(labels, size):
    format = np.zeros((len(labels), size))
    for i, label in enumerate(labels):
        format[i][label] = 1
    return format

def splitData(images, labels, ratio):
    temp = int(len(images) * ratio)
    return images[:temp], images[temp:], labels[:temp], labels[temp:]

def checkAccuracy(lst1, lst2):
    result = []
    for a,b in zip(lst1,lst2):
        if a == b:
            result.append(1)
        else:
            result.append(0)
    return np.mean(np.array(result))


def loadFashionMNIST():
    from tensorflow.keras.datasets import fashion_mnist

    (trainingImages, trainingLabels), (testingImages, testingLabels) = fashion_mnist.load_data()

    trainingImages = trainingImages.reshape(-1, 28 * 28) / 255.0
    testingImages = testingImages.reshape(-1, 28 * 28) / 255.0
    trainingLabels = convertIntoFormat(trainingLabels, 10)
    testingLabels = convertIntoFormat(testingLabels, 10)

    return trainingImages, testingImages, trainingLabels, testingLabels

def loadMNIST():
    from tensorflow.keras.datasets import mnist

    (trainingImages, trainingLabels), (testingImages, testingLabels) = mnist.load_data()

    trainingImages = trainingImages.reshape(-1, 28 * 28) / 255.0
    testingImages = testingImages.reshape(-1, 28 * 28) / 255.0
    trainingLabels = convertIntoFormat(trainingLabels, 10)
    testingLabels = convertIntoFormat(testingLabels, 10)

    return trainingImages, testingImages, trainingLabels, testingLabels


def showImg(flat_image, label=None):
    image = flat_image.reshape(28, 28)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    if label != None:
        plt.title("label: {label}")
    else:
        plt.title("Image:")
    
    plt.show()

def hyperparameterChanges(variable, start = 0, end = 0, step = 0, start2 = None, end2 = None, step2 = None):
    resultList = []
    if variable == 'lr':
        learningR = start + step
        epochs = 3
        while learningR < end:
            mlp = MLP(LAYERS,'sigmoid',maxAllowableOutput,'xavier')
            mlp.train(trainingImages,trainingLabels,epochs,learningR,batchSize,testingImages,True,0.95,False,1000,False)
            results = mlp.predict(testingImages)
            resultList.append(checkAccuracy(results,answers))
            learningR += step
        plotMovingAverage([resultList],2,step)
    if variable == 'optimizers':
            epochs = 3
            mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he')
            loss = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,True,0.95,True,1000,False)
            resultList.append(loss)
            mlp = MLP(LAYERS,'relu',maxAllowableOutput,'None')
            loss = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,True,0.95,True,1000,False)
            resultList.append(loss)
            print(resultList)
            plotMovingAverage(resultList,2)
    if variable == 'epochs':
        epochs = end
        mlp = MLP(LAYERS,'sigmoid',maxAllowableOutput,'xavier')
        losslist = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,True,0.95,True,False)
        print(losslist)
        plotMovingAverage([losslist],2,step)
    
    if variable == 'overfitting':
        epochs = end
        mlp = MLP(LAYERS,'sigmoid',maxAllowableOutput,'xavier')
        losslist,trainingLossList  = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,False,0.95,True,False,True)
        plotMovingAverage([losslist,trainingLossList],2)

    if variable == 'size':
        layerN = start2
        epochs = 10
        while layerN < end2:
            nodeSizes = start
            accuracyList = []
            while nodeSizes < end:
                layers = [784]
                for i in range (layerN):
                    layers.append(nodeSizes)
                layers.append(10)
                mlp = MLP(layers,'sigmoid',maxAllowableOutput,'xavier')
                mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels)
                results = mlp.predict(testingImages)
                accuracy = checkAccuracy(results,answers)
                accuracyList.append(accuracy)
                nodeSizes += int(step+(nodeSizes * 10 / end)*step)
                
            
            print(accuracyList)
            plotMovingAverage([accuracyList],2)
            print(f"Max accuracy at {start + step * accuracyList.index(max(accuracyList))} layer size and {layerN} layers")
            layerN += step2

    if variable == 'EfficiencyFrontier':

        #This randomly broke half way through I need to fix 
        width = start
        epochs = 100
        flopResults = []
        for layerN in range (start2,end2,step2):
            while width < end:
                layers = [784]
                for i in range (layerN):
                    layers.append(width)
                layers.append(10)
                mlp = MLP(layers,'sigmoid',maxAllowableOutput,'xavier')
                flopList = mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,True,0.95,True,True,False)
                print(flopList)
                flopResults.append(flopList)
                width += step
        print(flopResults)
        plotPolynomialWithTwo(flopResults,True)

    if variable == 'distribution':
        totalNodes = start
        ratio = step + start2
        epochs = 1
        for ratio in range (start2 + step, end, step):
            layers = [784,int(totalNodes * ratio),int(totalNodes * (1 - ratio)),10]
            mlp = MLP(layers,'sigmoid',maxAllowableOutput,'xavier')
            mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,True,0.95,False,1000,False)
            results = mlp.predict(testingImages)
            resultList.append(checkAccuracy(results,answers))
            print(ratio)

        print(resultList)
        plotMovingAverage([resultList],5)

    if variable == 'critBatch':
        pass

    return None


def loadCIFAR():

    #Change number of input nodes to 3072
    LAYERS[0] = 3072

    from tensorflow.keras.datasets import cifar10

    (trainingImages, trainingLabels), (testingImages, testingLabels) = cifar10.load_data()

    trainingImages = trainingImages.reshape(-1, 32 * 32 * 3) / 255.0
    testingImages = testingImages.reshape(-1, 32 * 32 * 3) / 255.0
    trainingLabels = convertIntoFormat(trainingLabels, 10)
    testingLabels = convertIntoFormat(testingLabels, 10)

    return trainingImages, testingImages, trainingLabels, testingLabels
 

def trainBestMNIST():
    #Best accuracy I got is 0.9852

    LAYERS = [784,575,10]       
    maxAllowableOutput = 500
    epochs = 16
    lr = 0.01
    batchSize = 50
    seed = 100

    mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he',seed)
    mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,True,0.9,False,False,False,True)
    return mlp

def trainBestFashionMNIST():
    #Best accuracy I got is 0.8967

    LAYERS = [784,64,10]       
    maxAllowableOutput = 500
    epochs = 30
    lr = 0.01
    batchSize = 50
    seed = 104

    mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he',seed)
    mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,True,0.9,False,False,False,True)
    return mlp

trainingImages, testingImages, trainingLabels, testingLabels = loadFashionMNIST()

saveFileName = 'SaveFile.txt'
LAYERS = [784,575,10]       
activationFunction = 'sigmoid'
maxAllowableOutput = 500
epochs = 15
lr = 0.005
batchSize = 50
traintestratio = 0.90  


# showImg(trainingImages[1])

#mlp = MLP(LAYERS,'relu',maxAllowableOutput,'he',100)
#mlp.train(trainingImages,trainingLabels,epochs,lr,batchSize,testingImages,testingLabels,True,0.9)

mlp = trainBestFashionMNIST()

answers = np.argmax(testingLabels, axis=1)

#hyperparameterChanges('optimizers')

#mlp = hyperparameterChanges('epochs',1,20)

#hyperparameterChanges('optimizers',0,0.1,0.01)

#hyperparameterChanges('size',400,1000,100,1,2,1)
results = mlp.predict(testingImages)
accuracy = checkAccuracy(results,answers)
print(f'Final accuracy from the test dataset: {accuracy:.4f} with {LAYERS} node layout')

#mlp.save(saveFileName)
#mlp.load(saveFileName)


#for i in range (len(testingImages)):
    #showImg(testingImages[i],testingLabels[i])

