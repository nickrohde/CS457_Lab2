# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:46:22 2018

@author: nick
"""

import random 

def main():
    random.seed()    
    
    # initial weights for 10 and 25 input neurons
    w_initial1 = [random.uniform(0,1) for _ in xrange(9)]
    w_initial2 = [random.uniform(0,1) for _ in xrange(25)]
    
    w_initial1.append(-1)
    w_initial2.append(-1)    
    
    lr_initial = 0.4
    
    # L = 1; I = -1
    # C = 1; U = -1
    # training data for 3x3 matrix with I and L
#    train1 = [[1,0,0, 1,0,0, 1,1,1, -1],
#              [0,1,0, 0,1,0, 0,1,0, -1]]
    # training data for 3x3 matrix with C and U
#    train2 = [[1,1,1, 1,0,0, 1,1,1, -1],
#              [1,0,1, 1,0,1, 1,1,1, -1]]
    # training data for 5x5 matrix with I and L
#    train3 = [[1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, -1],
#              [1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1, -1]]
    # training data for 5x5 matrix with C and U
#    train4 = [[1,1,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1, -1],
#              [1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 1,1,1,1,1, -1]]

    train1 = [[1,0,0, 1,1,0, 1,1,1, -1],
              [0,1,-1, 0,1,0, 0,0,0, -1]]
    # training data for 3x3 matrix with C and U
    train2 = [[1,1,1, 1,0,1, 1,1,1, -1],
              [1,0,1, 1,0,1, 1,0,1, -1]]
    # training data for 5x5 matrix with I and L
    train3 = [[1,0,0,1,0, 1,0,1,0,0, 1,0,0,0,1, 1,0,0,0,0, 1,1,0,0,0, -1],
              [1,0,0,0,1, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,0,1,1, -1]]
    # training data for 5x5 matrix with C and U
    train4 = [[1,1,0,1,1, 1,0,0,0,0, 1,0,1,0,0, 1,0,0,0,0, 1,1,0,1,1, 1],
              [1,0,0,0,1, 1,0,1,0,1, 1,0,0,0,1, 1,1,1,1,1, 1,1,1,1,0, -1]]


    targets1 = [1,-1]
    targets2 = [1,-1]
    targets3 = [-1,1]
    targets4 = [1,-1]

    # test vectors
    test1 = [[0,0,1, 0,0,1, 0,0,1, -1], [0,1,0, 0,1,0, 0,1,1, -1],
             [1,0,0, 1,0,0, 1,0,0, -1], [0,1,0, 0,1,0, 1,1,0, -1]]
             
    test2 = [[1,1,1, 0,0,1, 1,1,1, -1], [1,1,0, 1,0,0, 1,1,0, -1],
             [0,0,0, 1,0,1, 1,1,1, -1], [1,1,1, 1,0,1, 1,0,1, -1]]
    
    test3 = [[0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, -1],
             [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,0,0, -1],
             [0,1,1,1,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0, -1],
             [1,1,0,0,0, 1,1,0,0,0, 1,1,0,0,0, 1,1,1,1,1, 1,1,1,1,1, -1]]
             
    test4 = [[1,1,1,1,1, 1,1,1,1,1, 1,1,0,0,0, 1,1,1,1,1, 1,1,1,1,1, -1],
             [1,1,1,0,0, 1,0,0,0,0, 1,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0, -1],
             [1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, -1],
             [1,0,1,0,0, 1,0,1,0,0, 1,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0, -1]]

    # test vectors with random bits altered 
#    test1 = [[0,0,1, 0,0,1, 0,1,1, -1], [0,1,0, 0,0,0, 0,1,1, -1],
#             [1,0,0, 1,0,0, 1,-1,0, -1], [0,1,0, 0,1,1, 1,1,0, -1]]
             
#    test2 = [[1,1,1, 0,0,1, 1,0,1, -1], [0,1,0, 1,0,0, 1,1,0, -1],
#             [0,0,0, 1,1,1, 1,1,1, -1], [0,1,1, 1,0,1, 1,0,1, -1]]
    
#    test3 = [[0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,1, 0,0,1,0,0, 0,0,1,0,0, -1],
#             [0,1,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,0,0, -1],
#             [0,1,1,1,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0, -1],
#             [1,1,0,0,0, 1,1,0,0,0, 1,1,0,0,1, 1,1,1,1,1, 1,1,1,1,1, -1]]
             
#    test4 = [[1,1,1,1,1, 1,1,1,1,1, 1,1,0,0,0, 1,0,1,1,1, 1,1,1,1,1, -1],
#             [1,1,1,0,0, 1,0,0,0,0, 1,1,1,0,0, 0,1,0,0,0, 0,0,0,0,0, -1],
#             [1,0,0,0,1, 1,0,0,0,1, 1,1,0,1,1, 0,0,0,0,0, 0,0,0,0,0, -1],
#             [1,0,1,0,0, 1,0,1,0,0, 1,1,1,0,0, 0,1,0,0,0, 0,0,0,0,0, -1]]
            
    expected1 = [-1, 1, -1, 1]
    expected2 = [ 1, 1, -1,-1]
    expected3 = [-1, 1, -1, 1]
    expected4 = [ 1, 1, -1,-1]

    num_iterations = 100
    overall_accuracy1 = 0.0
    overall_accuracy2 = 0.0
    overall_accuracy3 = 0.0
    overall_accuracy4 = 0.0
    adjusted1 = 0.0
    adjusted2 = 0.0
    adjusted3 = 0.0
    adjusted4 = 0.0
    
    for i in range(num_iterations):    
        neuron1 = Neuron(w_initial1, lr_initial)
        neuron2 = Neuron(w_initial1, lr_initial)
        neuron3 = Neuron(w_initial2, lr_initial)
        neuron4 = Neuron(w_initial2, lr_initial)
    
        # train with train vectors
        neuron1.trainNeuron(train1,targets1)
        neuron2.trainNeuron(train2,targets2)
        neuron3.trainNeuron(train3,targets3)
        neuron4.trainNeuron(train4,targets4)

        # train with test vectors        
#        neuron1.trainNeuron(test1,expected1)
#        neuron2.trainNeuron(test2,expected2)
#        neuron3.trainNeuron(test3,expected3)
#        neuron4.trainNeuron(test4,expected4)

        adjusted1 += neuron1.changes
        adjusted2 += neuron2.changes
        adjusted3 += neuron3.changes
        adjusted4 += neuron4.changes

        # validate with test vectors
        overall_accuracy1 += printResults(test1,expected1,neuron1)
        overall_accuracy2 += printResults(test2,expected2,neuron2)
        overall_accuracy3 += printResults(test3,expected3,neuron3)
        overall_accuracy4 += printResults(test4,expected4,neuron4)
        
        # validate with train vectors
#        overall_accuracy1 += printResults(train1,targets1,neuron1)
#        overall_accuracy2 += printResults(train2,targets2,neuron2)
#        overall_accuracy3 += printResults(train3,targets3,neuron3)
#        overall_accuracy4 += printResults(train4,targets4,neuron4)
        
    overall_accuracy1 /= float(num_iterations)
    overall_accuracy2 /= float(num_iterations)
    overall_accuracy3 /= float(num_iterations)
    overall_accuracy4 /= float(num_iterations)

    adjusted1 /= float(num_iterations)
    adjusted2 /= float(num_iterations)
    adjusted3 /= float(num_iterations)
    adjusted4 /= float(num_iterations)
    
    print "Average accuracy of neuron 1:", overall_accuracy1, "%"
    print "Average accuracy of neuron 2:", overall_accuracy2, "%"
    print "Average accuracy of neuron 2:", overall_accuracy3, "%"
    print "Average accuracy of neuron 3:", overall_accuracy4, "%\n"

    print "Average adjustements to neuron 1:", adjusted1
    print "Average adjustements to neuron 2:", adjusted2
    print "Average adjustements to neuron 3:", adjusted3
    print "Average adjustements to neuron 3:", adjusted4
    
def printResults(x,t,neuron):
    file = open("results", "a")
    correct = 0.0
    accuracy = 0.0
    
    for i in range(len(x)):
        y = neuron.evaluate(x[i])
        file.write("Neuron evaulated to ")
        file.write(str(y))
        file.write(" expected ")
        file.write(str(t[i]))
        file.write("\n")
        if y == t[i]:
            correct += 1.0

    accuracy = (correct / len(x)) * 100.0
    
    file.write("Accuracy: ")
    file.write(str(accuracy))
    file.write("%\n")
    file.write("Final weights of neuron:\n")
    file.write(" ".join('%0.2f' % item for item in neuron.w))
    file.write("\n\n")

    return accuracy
    
class Neuron:
    w = []
    LEARN_RATE = -1

    def __init__(self, w_initial, learnRate):
        self.LEARN_RATE = learnRate
        self.w = list(w_initial)
        self.changes = 0
    
    def evaluate(self, x):
        y = 0
        for i in range(len(x)):
            y += self.w[i] * x[i]
        
        if y > 0: 
            return 1
        else:
            return -1
        
    def trainNeuron(self, x, t):
        successful = False
        
        # train until all training vectors were evaluated correctly
        while not successful:
            successful = True

            # shuffle the order of the training set
            order = random.sample(range(len(x)), len(x))

            for i in order:
                y = self.evaluate(x[i])

                # neuron evaluated correctly, do nothing
                if (y == t[i]):
                    continue
                    
                # neuron failed, adjust weights
                else:
                    successful = False
                    self.adjustWeights(x[i],y,t[i])
        
    def adjustWeights(self, x, y, t):
        self.changes += 1
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.LEARN_RATE * (y-t) * x[i]
                        
main()
