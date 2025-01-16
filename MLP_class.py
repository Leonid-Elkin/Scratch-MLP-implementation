import numpy as np

class MLP:
    def __init__(self, layers, activation, clipValue, weightInit, seed=None):
        np.random.seed(seed)
        self.numLayers = len(layers)
        self.layerSizes = layers
        self.weights = []
        self.biases = []
        self.activationFunction = activation
        self.clipValue = clipValue
        self.weightInit = weightInit.lower()
        self.CPUUsage = 0

        for layer in range(self.numLayers - 1):
            if self.weightInit == "xavier":
                weight = np.random.randn(layers[layer], layers[layer + 1]) / np.sqrt(layers[layer])
            elif self.weightInit == "he":
                weight = np.random.randn(layers[layer], layers[layer + 1]) * np.sqrt(2 / layers[layer])
            else:
                weight = np.random.randn(layers[layer], layers[layer + 1])

            self.weights.append(weight)
            self.biases.append(np.zeros((1, layers[layer + 1])))

    def sigmoid(self, x):
        x = np.clip(x, -self.clipValue, self.clipValue)
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        x = np.clip(x, -self.clipValue, self.clipValue)
        return np.maximum(0, x)

    def reluDerivative(self, x):
        return np.where(x > 0, 1, 0)

    def leakyRelu(self, x, a = 0.01):
        x = np.clip(x, -self.clipValue, self.clipValue)
        return np.where(x > 0, x, a * x)

    def leakyReluDerivative(self, x, a=0.01):
        return np.where(x > 0, 1, a)

    def softmax(self, x):
        x = np.clip(x, -self.clipValue, self.clipValue)
        expon = np.exp(x - np.max(x, axis=1, keepdims=True))
        return expon / np.sum(expon, axis=1, keepdims=True)

    def activation(self, x):
        if self.activationFunction == 'sigmoid':
            return self.sigmoid(x)
        elif self.activationFunction == 'relu':
            return self.relu(x)
        elif self.activationFunction == 'leakyrelu':
            return self.leakyRelu(x)

    def activationDerivative(self, x):
        if self.activationFunction == 'sigmoid':
            return self.sigmoidDerivative(x)
        elif self.activationFunction == 'relu':
            return self.reluDerivative(x)
        elif self.activationFunction == 'leakyrelu':
            return self.leakyReluDerivative(x)

    def forward(self, incomingConnections, FLOPPER=False):
        self.activations = [incomingConnections]
        for layer in range(self.numLayers - 2):
            dotProduct = np.dot(self.activations[-1], self.weights[layer]) + self.biases[layer]
            activatedDotProduct = self.activation(dotProduct)
            self.activations.append(activatedDotProduct)

            if FLOPPER:
                height, width = self.activations[-2].shape
                randomThing = self.weights[-1].shape[1]
                flops = 2 * height * width * randomThing
                if self.activationFunction == 'sigmoid':
                    flops *= 4
                
                self.CPUUsage += flops

        outputLayer = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        answer = self.softmax(outputLayer)
        self.activations.append(answer)

        if FLOPPER:
            height, width = self.activations[-2].shape
            randomThing = self.weights[-1].shape[1]
            flops = 2 * height * width * randomThing
            if self.activationFunction == 'sigmoid':
                flops *= 4
            
            self.CPUUsage += flops

        return answer

    def crossEntropyLoss(self, predictions, labels):
        randomSmallNumber = 1e-12
        predictions = np.clip(predictions, randomSmallNumber, 1.0 - randomSmallNumber)
        loss = -np.sum(labels * np.log(predictions)) / labels.shape[0]
        return loss

    def backward(self, targets, lr, FLOPPER):
        outputLayerError = self.activations[-1] - targets

        errorO = [outputLayerError]
        for index in range(self.numLayers - 2, 0, -1):
            error = np.dot(errorO[0], self.weights[index].T) * self.activationDerivative(self.activations[index])
            errorO.insert(0, error)

            if FLOPPER:
                height, width = self.activations[-2].shape
                randomThing = self.weights[-1].shape[1]
                flops = 2 * height * width * randomThing
                if self.activationFunction == 'sigmoid':
                    flops *= 4
                
                self.CPUUsage += flops

        for index in range(self.numLayers - 1):
            weightGradient = np.dot(self.activations[index].T, errorO[index])
            biasGradient = np.sum(errorO[index], axis=0, keepdims=True)

            self.weights[index] -= lr * weightGradient
            self.biases[index] -= lr * biasGradient

            if FLOPPER:
                height, width = self.activations[-2].shape
                randomThing = self.weights[-1].shape[1]
                flops = 2 * height * width * randomThing
                if self.activationFunction == 'sigmoid':
                    flops *= 4
                
                self.CPUUsage += flops

            # This apparently works even though its everything just timed together

    def train(self, inputs, targets, numEpochs, lr, batchSize, testInputs, testLabels, lrDecay=False, decayRate=0.95, avgLossToggle=False, recordUsage=False, epochLosses=False):
        indexes = inputs.shape[0]
        lossList = []
        usageSuperlist = []
        epochLossesList = []

        for epoch in range(numEpochs):
            lossEpochList = []
            shuffledIndexes = np.random.permutation(indexes)
            Inputs = inputs[shuffledIndexes]
            Labels = targets[shuffledIndexes]

            if epochLosses:
                testPredictions = self.forward(testInputs)
                testLoss = self.crossEntropyLoss(testPredictions, testLabels)
                epochLossesList.append(testLoss)
                print(testLoss)

            for batch in range(0, indexes, batchSize):
                batchInputs = Inputs[batch: (batch + batchSize)]
                batchLabels = Labels[batch: (batch + batchSize)]

                predictions = self.forward(batchInputs, recordUsage)
                loss = self.crossEntropyLoss(predictions, batchLabels)
                lossEpochList.append(loss)

                self.backward(batchLabels, lr, recordUsage)

            if lrDecay:
                lr *= decayRate
            avgLoss = np.mean(lossEpochList)


            if recordUsage:
                testPredictions = self.forward(testInputs)
                avgLoss = self.crossEntropyLoss(testPredictions, testLabels)
                usageSuperlist.append([avgLoss, self.CPUUsage])
    

            if avgLossToggle:
                lossList.append(avgLoss)

            print(f'Epoch {epoch + 1} / {numEpochs}, Average Loss: {avgLoss:.10f}, CPU FLOPs: {self.CPUUsage} lr: {lr:.10f}')

        if recordUsage:
            return usageSuperlist

        if epochLosses:
            if avgLoss:
                return lossList, epochLossesList
            else:
                return epochLossesList
        else:
            if avgLossToggle:
                return lossList

    def predict(self, inputs):
        predictions = self.forward(inputs)
        return np.argmax(predictions, axis=1)

    def save(self, filename):
        import pickle
        file = open(filename, 'wb')
        pickle.dump({
            'layerSizes': self.layerSizes,
            'weights': self.weights,
            'biases': self.biases,
            'activationFunction': self.activationFunction,
            'clipValue': self.clipValue,
            'weightInit': self.weightInit
        }, file)
        file.close()

    def load(self, filename):
        import pickle
        file = open(filename,'rb')
        data = pickle.load(file)
        self.layerSizes = data['layerSizes']
        self.numLayers = len(self.layerSizes)
        self.weights = data['weights']
        self.biases = data['biases']
        self.activationFunction = data['activationFunction']
        self.clipValue = data['clipValue']
        self.weightInit = data['weightInit']
        file.close()

