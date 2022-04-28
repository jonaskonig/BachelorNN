# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import NeuralNetwork
import Communicator
import Manager
import numpy as np
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    neuralnet = [[4, 3, 4], [4, 3, 4], [4, 3, 4], [12, 4, 5, 3, 2]]
    neuralneta = []
    botsize = 15
    port = 9653
   # mana = Manager.CommunicationManager(port, "127.0.0.1", botsize)
   # port += 4
    #time.sleep(0.5)
    #mana.setshuffle()
    #mana.setstart()
    for x in range(botsize):
        neuro = NeuralNetwork.NeuralNet(neuralnet, upperbound=1, lowerbound=-1)
        neuro.initvalues()
        neuralneta.append(neuro)
    #t = neuralneta[0].createoppositeindividual(1,-1)
    #print(len(t.getencoded()))
    cendobl = NeuralNetwork.CENDEDOBL(neuralneta,0.3,30,neuralnet,port,15)
    cendobl.CenDEDOL(0.9,0.5,3)
    #time.sleep(30)
   # print("stopping")
    #mana.setstop()
    #for x in neuralneta:
     #   x.setstoprunning()
    #for x in neuralneta:
    #    x.stopthread()
    #results = mana.getresults()
    #while results is None:
    #    results = mana.getresults()
    #print(results)
    # time.sleep(1)
    # mana.setstop()
    # print("hallo")
    # comm = Communicator.Communicator(8653, "127.0.0.1", nn)
    # time.sleep(40)
    # del comm
    # input_data = np.random.rand(3,5,4).tolist()
    # start = time.time()
    # out =nn.neuralnet(input_data)
# end = time.time()
# print(out)
# print("The time of execution of above program is :", end - start)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
