import random
import math
import copy

def sigmoid(x):
    return 1/(1+math.pi**x)

def identity(x):
    return x

def generateNeuralNetwork(nInput,wHidden,hHidden,nOutput,activationFunc):
    neural_network = {}
    neural_network["caracteristic"] = {"nInput":nInput,
                                       "nOutput":nOutput,
                                       "hiddenLayer":wHidden,
                                       "nodePerLayer":hHidden
                                       }
    neural_network["node"] = []

    for l in range(wHidden+1):
        node = {"bias":0,
                "activation_function":activationFunc,
                "value_holder":0
                }
        if l==0:
            node["weigth"] = [0 for i in range(nInput)]
        else:
            node["weigth"] = [0 for i in range(hHidden)]

        if l==wHidden:
            neural_network["node"].append([copy.deepcopy(node) for i in range(nOutput)])
        else:
            neural_network["node"].append([copy.deepcopy(node) for i in range(hHidden)])
    
    return neural_network

def mutate(neural_network,mutationRange):
    new_neural_network = copy.deepcopy(neural_network)
    
    for l in range(new_neural_network["caracteristic"]["hiddenLayer"]+1):
        for n in range(len(new_neural_network["node"][l])):

            new_neural_network["node"][l][n]["bias"] += mutationRange*random.random() - mutationRange/2
            
            for w in range(len(new_neural_network["node"][l][n]["weigth"])):
                new_neural_network["node"][l][n]["weigth"][w] += mutationRange*random.random() - mutationRange/2

    return new_neural_network

def generate_child(neural_network,parents,mutationRange):
    new_neural_network = neural_network.copy()

    for l in range(new_neural_network["caracteristic"]["hiddenLayer"]+1):
        for n in range(len(new_neural_network["node"][l])):

            new_neural_network["node"][l][n]["bias"] = random.choice(parents)["node"][l][n]["bias"] + (mutationRange*random.random() - mutationRange/2)

            for w in range(len(new_neural_network["node"][l][n]["weigth"])):
                new_neural_network["node"][l][n]["weigth"][w] = random.choice(parents)["node"][l][n]["weigth"][w] + (mutationRange*random.random() - mutationRange/2)

    return new_neural_network

def returnValue(neural_network,inputVal):
    for l in range(neural_network["caracteristic"]["hiddenLayer"]+1):
        for n in range(len(neural_network["node"][l])):
            val = 0

            for w in range(len(neural_network["node"][l][n]["weigth"])):
                if l != 0:
                    val += neural_network["node"][l][n]["weigth"][w] * neural_network["node"][l-1][w]["value_holder"]
                else:
                    val += neural_network["node"][l][n]["weigth"][w] * inputVal[w]

            val += neural_network["node"][l][n]["bias"]
            neural_network["node"][l][n]["value_holder"] = neural_network["node"][l][n]["activation_function"](val)

    return [neural_network["node"][neural_network["caracteristic"]["hiddenLayer"]][i]["value_holder"] for i in range(neural_network["caracteristic"]["nOutput"])]
