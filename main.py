# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import NeuralNetwork
import Communicator
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    neuralnet = [[4,3,4],[4,3,4],[4,3,4],[12,4,5,1,2]]
    nn = NeuralNetwork.NeuralNet(neuralnet)
    nn.initvalues()
    input_data = np.random.rand(3,5,4).tolist()
    out =nn.neuralnet(input_data)
    print(out)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
