import json
import random
import math
import sys
import threading
import time

import numpy as np
from typing import List
import Communicator
import Manager


class NeuralNet:
    # initiallise a neural net. Layer beeing an array, where every position indicates how many neurons this layer has
    # the first and last position of the list are the input and output layers of the neural net
    def __init__(self, layer: List[List], encodednet=np.empty(0), upperbound=1, lowerbound=-1, performance=None):
        self.layer = layer
        self.Neuralnetlengh = self.caculatelnght()
        self.offset = self.calcoffset()
        self.performance = performance
        self.neuron = []

        self.encodednet = encodednet
        self.endsofnns = self.setends()
        if len(encodednet) > 0:
            self.calculatelowerandupperbound()
        else:
            self.upperbound = upperbound
            self.lowerbound = lowerbound

    def caculatelnght(self):
        count = 0
        for x in self.layer:
            neuroncount = sum(x)
            count += neuroncount + pow(neuroncount, 2) - x[0]
        return count

    def calculatelowerandupperbound(self):
        self.lowerbound = np.min(self.encodednet)
        self.upperbound = np.max(self.encodednet)

    def initvalues(self):
        lenngh = 0
        for nn in self.layer:
            lenngh += sum(nn[1:])
            for i in range(len(nn)):
                if i + 1 == len(nn):
                    continue
                lenngh += nn[i] * nn[i + 1]
        self.encodednet = np.random.uniform(low=self.lowerbound, high=self.upperbound, size=lenngh)
        self.calculatelowerandupperbound()
        print(len(self.encodednet))

    def setends(self):
        begins = []
        for nn in self.layer:
            lenngh = 0
            lenngh += sum(nn[1:])
            for i in range(len(nn)):
                if i + 1 == len(nn):
                    continue
                lenngh += nn[i] * nn[i + 1]
            if len(begins) > 0:
                begins.append(begins[-1] + lenngh)
            else:
                begins.append(lenngh)
        return begins

    def calcoffset(self):
        count = sum(self.layer[0])
        return count + pow(count, 2) - self.layer[0][0]

    def getperformance(self):
        return self.performance

    def getupperbound(self):
        return self.upperbound

    def getlowerbound(self):
        return self.lowerbound

    def getencoded(self):
        return self.encodednet

    def setencoded(self, encoded):
        self.encodednet = encoded

    def setperfromance(self, performance):
        self.performance = performance

    # reduces the deminesion of the neual network
    # to allow tweeking of importance from differnt demensions, these neurons have the possiblility to tweek the bias
    # for each demension, while the weight is staying the same
    def reducedeminesion(self, input, bias):
        out = 0
        for x, y in zip(input, bias):
            out += input * bias
        return out

    def createoppositeindividual(self, upper, lower):
        oppositeencodednet = np.empty(0)
        for x in self.encodednet:
            value = random.uniform((upper + lower) / 2, lower + upper - x)
            oppositeencodednet = np.append(oppositeencodednet, value)
        return NeuralNet(self.layer, oppositeencodednet, upperbound=self.lowerbound, lowerbound=self.lowerbound)

    def feedforward(self, input, layer: List, index, offset=0):
        neuron = []
        for x in input:
            neuron.append(x)
        if offset == 0:
            weightcounter = 0
        else:
            weightcounter = self.endsofnns[offset - 1] + 1
        biascounter = self.endsofnns[offset] - sum(layer[1:])
        neuroncounter = 0
        for x in range(len(layer)):
            value = 0
            if x + 1 < len(layer):
                for i in range(layer[x + 1]):
                    for t in range(layer[x]):
                        value += neuron[neuroncounter] * self.encodednet[weightcounter]
                        weightcounter += 1
                neuroncounter += 1
                neuron.append(math.tanh(value + self.encodednet[biascounter]))
                biascounter += 1
        self.neuron[index] = np.array(neuron[-layer[-1]:])

    def neuralnet(self, input: List[List]):
        threds = list()
        index = 0
        nnposition = 0
        self.neuron = []
        for p in input:
            for t in p:
                self.neuron.append(0)
                x = threading.Thread(target=self.feedforward, args=(t, self.layer[nnposition], index, nnposition))
                index += 1
                x.start()
                threds.append(x)
            nnposition += 1
        for t in threds:
            t.join()
        newinput = np.empty(0)
        inputcounter = 0
        for x in input:
            temparray = np.empty(len(self.neuron[0]))
            for t in range(inputcounter, len(x)):
                temparray += self.neuron[t]
            inputcounter += len(x) - 1
            newinput = np.concatenate((newinput, temparray))
        self.feedforward(newinput, self.layer[nnposition], 0, nnposition)
        return self.neuron[0]


class CENDEDOBL:
    def __init__(self, populationsize: List[NeuralNet], jumpingrate, runtime, layer, startport, chunksize,
                 save: str = "./",
                 address="127.0.0.1"):
        self.populationsize = populationsize
        self.jumpingrate = jumpingrate
        self.lowerbound = 1
        self.runtime = runtime
        self.upperbound = -1
        self.startport = startport
        self.layer = layer
        self.save = save
        self.chunksize = chunksize
        self.iteration = 0
        self.address = address
        self.manager = Manager.CommunicationManager(self.startport, self.address, self.chunksize)
        self.manager.setshuffle()
        self.startport += 4

    def writedata(self):
        data = {"layer": self.layer,
                'jumpingrate': self.jumpingrate,
                "lowerbound": self.lowerbound,
                "upperbound": self.upperbound
                }
        nets = []
        performance = []
        boundries = []
        for x in self.populationsize:
            nets.append(x.getencoded().tolist())
            performance.append(x.getperformance())
            boundries.append([x.getupperbound(), x.getlowerbound()])
        data["individuals"] = nets
        data["performance"] = performance
        data["boundries"] = boundries
        with open(self.save + str(self.iteration)+".json", "w") as outfile:
            json.dump(data, outfile)
        self.iteration += 1

    def readindata(self, filename: str):
        data = json.load(open(filename))
        self.layer = data['layer']
        self.jumpingrate = data['jumpingrate']
        self.lowerbound = data['lowerbound']
        self.upperbound = data['upperbound']
        performance = data["performance"]
        individual = data["individuals"]
        boundries = data["boundries"]
        self.populationsize = []

        for x in range(len(individual)):
            self.populationsize.append(
                NeuralNet(self.layer, np.array(individual[x]), upperbound=boundries[x][0], lowerbound=boundries[x][1], performance=performance[x]))

    def lowerandupperbound(self):
        for x in self.populationsize:
            self.upperbound = x.getupperbound() if x.getupperbound() > self.upperbound else self.upperbound
            self.lowerbound = x.getlowerbound() if x.getlowerbound() < self.lowerbound else self.lowerbound
        print(f"self upper {self.upperbound} self lower {self.lowerbound}")

    def obl(self):
        self.lowerandupperbound()
        o_pop: List[NeuralNet] = []
        for x in self.populationsize:
            o_pop.append(x.createoppositeindividual(self.lowerbound, self.upperbound))
        self.findbestindividuals(o_pop)

    def sortfunc(self, neuralnet: NeuralNet):
        return NeuralNet.getperformance(neuralnet)

    def findbestindividuals(self, coparer: List[NeuralNet], onebyone=False):
        counter = 0
        while counter < len(coparer):
            startport = self.startport
            currentactiveingame = []
            if len(coparer) - counter > self.chunksize:
                self.manager.setbotcount(self.chunksize)
                self.manager.setstart()
                for x in range(self.chunksize):
                    currentactiveingame.append(Communicator.Communicator(startport, self.address, coparer[counter]))
                    counter += 1
                    startport += 4
            else:
                self.manager.setbotcount(len(coparer) - counter)
                self.manager.setstart()
                for x in range(len(coparer) - counter):
                    currentactiveingame.append(Communicator.Communicator(startport, self.address, coparer[counter]))
                    counter += 1
                    startport += 4
            time.sleep(self.runtime)
            self.manager.setstop()
            print("setstop")
            for active in currentactiveingame:
                active.setstoprunning()
            for active in currentactiveingame:
                active.stopthread()
            results = self.manager.getresults()
            while results is None:
                #     self.manager.askforresult()
                print("Waitingforresults")
                #    time.sleep(1)
                results = self.manager.getresults()
            print(results)
            x = 0
            for res in range(len(results), 0, -1):
                coparer[counter - res].setperfromance(results[x])
                x += 1
        if onebyone:
            for x in range(len(self.populationsize) - 1):
                self.populationsize[x] = coparer[x] if coparer[x].getperformance() > self.populationsize[
                    x].getperformance() else self.populationsize[x]
        else:
            print("comparing")
            orgleng = len(self.populationsize)
            self.populationsize.extend(coparer)
            self.populationsize.sort(key=self.sortfunc, reverse=True)
            self.populationsize = self.populationsize[:orgleng]
        time.sleep(1)

    def evaluateindividum(self, individum: NeuralNet):
        self.manager.setbotcount(1)
        self.manager.setstart()
        t = Communicator.Communicator(self.startport, self.address, individum)
        time.sleep(self.runtime)
        self.manager.setstop()
        print("setstop")
        t.setstoprunning()
        t.stopthread()
        t.running = False
        del t
        results = False
        while not results:
            print("waiting for results")
            results = self.manager.checkresults()
        results = self.manager.getresults()
        while results is None:
            #     self.manager.askforresult()
            print("Waitingforresults")
            #    time.sleep(1)
            results = self.manager.getresults()
        print(results)
        individum.setperfromance(results[0])
        self.populationsize.append(individum)
        self.populationsize.sort(key=self.sortfunc, reverse=True)
        del self.populationsize[-1]
        time.sleep(1)

    def CenDEDOL(self, crossoverrate, scalingfactor: float, bestsolutions):
        print(len(self.populationsize[0].getencoded()))
        self.findbestindividuals(self.populationsize)
        print(len(self.populationsize[0].getencoded()))
        self.obl()
        print(len(self.populationsize[0].getencoded()))
        while True:
            self.lowerandupperbound()
            self.writedata()
            self.manager.setshuffle()
            x1 = random.randint(0, len(self.populationsize) - 1)
            x2 = random.randint(0, len(self.populationsize) - 1)
            while x1 == x2:
                x1 = random.randint(0, len(self.populationsize) - 1)
                x2 = random.randint(0, len(self.populationsize) - 1)
            scaledpopulation: List[NeuralNet] = []
            self.lowerandupperbound()
            for x in self.populationsize:
                scal1 = (scalingfactor * (
                        self.populationsize[0].getencoded() - x.getencoded()))
                scal2 = (scalingfactor * (
                        self.populationsize[x1].getencoded() - self.populationsize[x2].getencoded()))
                newencoded = x.getencoded() + scal1 + scal2

                index = 0
                for t in x.getencoded():
                    trand = random.random()
                    if not trand < crossoverrate or trand == t:
                        newencoded[index] = t
                    index += 1

                scaledpopulation.append(NeuralNet(self.layer, newencoded, self.upperbound, self.lowerbound))
            self.findbestindividuals(scaledpopulation, True)
            if random.random() < self.jumpingrate:
                self.obl()
            else:
                self.lowerandupperbound()
                t = NeuralNet(self.layer, upperbound=self.upperbound, lowerbound=self.lowerbound)
                t.initvalues()
                net = t.getencoded()
                for i in range(bestsolutions):
                    net += self.populationsize[i].getencoded()
                t = NeuralNet(self.layer, net / bestsolutions, upperbound=self.upperbound, lowerbound=self.lowerbound)
                self.evaluateindividum(t)
