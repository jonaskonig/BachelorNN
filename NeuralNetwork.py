import json
import random
import math
import sys
import numpy as np
from typing import List




class NeuralNet:
    # initiallise a neural net. Layer beeing an array, where every position indicates how many neurons this layer has
    # the first and last position of the list are the input and output layers of the neural net
    def __init__(self, layer, encodednet=np.empty(), upperbound=sys.maxsize, lowerbound=-1*sys.maxsize, boundries = np.empty(2)):
        self.layer = layer

        self.performance = 0
        self.neuron = []
        self.boundries = boundries
        self.encodednet = encodednet
        self.upperbound = upperbound
        self.lowerbound = lowerbound

    def getperformance(self):
        return self.performance
    def getupperbound(self):
        return self.upperbound

    def getlowerbound(self):
        return self.lowerbound

    def getencoded(self):
        return self.encodednet
    def getperformance(self):
        return self.performance

    def setencoded(self, encoded):
        self.encodednet = encoded
    def setperfromance(self, performance):
        self.performance = performance

    def initNeurons(self):
        neuroncount = sum(self.layer)
        numberofweights = pow(neuroncount, 2)
        for x in range(neuroncount + numberofweights - self.layer[0]):
            value = random.uniform(self.boundries[0], self.boundries[1])
            self.upperbound = value if value > self.upperbound else None
            self.lowerbound = value if value < self.lowerbound else None
            self.encodednet = np.append(self.encodednet, value)

    def createoppositeindividual(self, upper, lower):
        oppositeencodednet = np.array()
        localupper = sys.minint
        locallower = sys.maxint
        neuroncount = sum(self.layer)
        numberofweights = pow(neuroncount, 2)
        for x in range(neuroncount + numberofweights - self.layer[0]):
            value = random.uniform((upper + lower) / 2, lower + upper - self.encodednet[x])
            locallower = value if value < locallower else None
            localupper = value if value > localupper else None
            oppositeencodednet = np.append(oppositeencodednet, value)
        return NeuralNet(self.layer, oppositeencodednet, localupper, locallower, self.boundries)

    def feedforward(self, input):
        for x in input:
            self.neuron.append(x)
        weightcounter = 0
        biascounter = pow(sum(self.layer), 2) - 1
        for x in self.layer:
            value = 0
            for i in range(next(x)):
                for t in range(x):
                    value += self.neuron[weightcounter] * self.encodednet[weightcounter]
                    weightcounter += 1
            self.neuron.append(math.tanh(value + self.encodednet[biascounter]))
            biascounter += 1


class CENDEDOBL:
    def __init__(self, populationsize: List[NeuralNet], jumpingrate, layer, save = "./",evaluationgfunc):
        self.populationsize = populationsize
        self.evaluationfuc = evaluationgfunc
        self.jumpingrate = jumpingrate
        self.lowerbound = sys.maxint
        self.upperbound = sys.minint
        self.layer = layer
        self.save = save
        self.iteration = 0

    def writedata(self):
        data = {"layer":self.layer,
                'jumpingrate':self.jumpingrate,
                "lowerbound":self.lowerbound,
                "upperbound":self.lowerbound
                }
        nets = []
        for x in self.populationsize:
            np.append(nets,x.getencoded())
        data["individuals"] = nets
        with open(self.save+str(self.iteration), "w") as outfile:
            json.dump(data, outfile)
        self.iteration+=1

    def lowerandupperbound(self):
        for x in self.populationsize:
            self.upperbound = x.getupperbound if x.getupperbound > self.upperbound else None
            self.lowerbound = x.getlowerbound if x.getlowerbound < self.lowerbound else None

    def obl(self):
        OPop: List[NeuralNet] = []
        for x in self.populationsize:
            OPop.append(x.createoppositeindividual(self.lowerbound,self.upperbound))
        self.findbestindividuals(self.OPop)

    def sortfunc(self,neuralnet: NeuralNet):
        return NeuralNet.getperformance(neuralnet)

    def findbestindividuals(self,coparer:List[NeuralNet]):
        out = self.evaluationfuc([*coparer,*self.populationsize])
        out.sort(key=self.sortfunc)
        self.populationsize = out[:len(self.populationsize)]

    def evaluateindividum(self, individum):
        out = self.evaluationfuc([individum])
        self.populationsize.append(out[0])
        self.populationsize.sort(key=self.sortfunc)
        del self.populationsize[-1]

    def CenDEDOL(self, crossoverrate, scalingfactor: float, bestsolutions):
        self.obl()
        while True:
            x1 = random.randint(0, len(self.populationsize))
            x2 = random.randint(0, len(self.populationsize))
            while x1 == x2:
                x1 = random.randint(0, len(self.populationsize))
                x2 = random.randint(0, len(self.populationsize))
            scaledpopulation: List[NeuralNet] = []
            for x in self.populationsize:
                newencoded = x.getencoded + scalingfactor * (self.populationsize[0].getencoded() - x.getencoded) + scalingfactor * (
                            self.populationsize[x1].getencoded() - self.populationsize[x2].getencoded())

                for index, t in x.getencoded:
                    trand = random.random()
                    if not trand < crossoverrate or trand == t:
                        newencoded[index] = t
                scaledpopulation.append(NeuralNet(self.layer,newencoded,self.upperbound,self.lowerbound))
            self.findbestindividuals(scaledpopulation)
            if random.random()< self.jumpingrate:
                self.obl()
            else:
                t = np.zeros(pow(sum(self.layer),2)+sum(self.layer))
                for i in range(bestsolutions): t += self.populationsize[i].getencoded()
                t = NeuralNet(self.layer,t,self.upperbound,self.lowerbound)
                self.evaluateindividum(t)


