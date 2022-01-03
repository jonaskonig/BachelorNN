import random
import math
class NeuralNet:
    def __init__(self,layer):
        self.layer = layer

    def initNeurons(self):
        self.weights = []
        self.bias = []
        for x in self.layer:
            numberofweights = x * next(self.layer)
            for k in range(numberofweights):
                self.weights.append(random.uniform(-0.5, 0.5))
            for p in range(x):
                self.bias.append(random.uniform(-0.5,0.5))


    def feedforward(self, input):
        self.neuron = []
        for x in input:
            self.neuron.append(x)
        weightcounter = 0
        biascounter = 0
        for x in self.layer:
            value = 0
            for i in range(next(x)):
                for t in range(x):
                    value += self.neuron[weightcounter]*self.weights[weightcounter]
                    weightcounter+=1
            self.neuron.append(math.tanh(value+self.bias[biascounter]))
            biascounter+=1





