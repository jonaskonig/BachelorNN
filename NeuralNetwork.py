import random
import math
import sys
class NeuralNet:
    #initiallise a neural net. Layer beeing an array, where every position indicates how many neurons this layer has
    #the first and last position of the list are the input and output layers of the neural net
    def __init__(self,layer, weights = [], bias = [], upperbound = sys.minint, lowerbound =  sys.maxint):
        self.layer = layer
        self.neuron = []
        self.weights = weights
        self.bias = bias
        self.upperbound = upperbound
        self.lowerbound = lowerbound

    def getupperbound(self):
        return self.upperbound
    def getlowerbound(self):
        return self.lowerbound


    def initNeurons(self):
        for x in self.layer:
            numberofweights = x * next(self.layer)
            for k in range(numberofweights):
                value = random.uniform(-0.5, 0.5)
                self.upperbound = value if value > self.upperbound else None
                self.lowerbound = value if value < self.lowerbound else None
                self.weights.append(value)
            for p in range(x):
                value = random.uniform(-0.5, 0.5)
                self.upperbound = value if value > self.upperbound else None
                self.lowerbound = value if value < self.lowerbound else None
                self.bias.append(value)

    def createoppositeindividual(self, upper, lower):
        oppositeweight = []
        oppositebias = []
        localupper = sys.minint
        locallower = sys.maxint
        for x in self.layer:
            numberofweights = x * next(self.layer)
            for k in range(numberofweights):
                value = random.uniform((upper+lower)/2,lower+upper-self.weights[k])
                locallower = value if value < locallower else None
                localupper = value if value > localupper else None
                oppositeweight.append(value)
            for p in range(x):
                value = random.uniform((upper+lower)/2,lower+upper-self.weights[p])
                locallower = value if value < locallower else None
                localupper = value if value > localupper else None
                oppositebias.append(value)
        return NeuralNet(self.layer,oppositeweight,oppositebias,localupper,locallower)
    def feedforward(self, input):
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

class CENDEDOBL:
    def __init__(self, populationsize: List[NeuralNet],jumpingrate):
        self.populationsize = populationsize
        self.jumpingrate = jumpingrate
        self.lowerbound = sys.maxint
        self.upperbound = sys.minint

    def lowerandupperbound(self):
        for x in self.populationsize:
            self.upperbound = x.getupperbound if x.getupperbound > self.upperbound else None
            self.lowerbound = x.getlowerbound if getlowerbound < self.lowerbound else None

    def obl(self):
        OPop: List[NeuralNet] = []
        for x in self.populationsize:
            OPop.append(x.createoppositeindividual())






