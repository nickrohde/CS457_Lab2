# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:46:22 2018

@author: nick
"""

import random 

def main():
    w_initial1 = [1,-1,-1,1,0,1,1,0,0,1]
    w_initial2 = [1,-1,-1,1,0,1,1,0,0,1,1,-1,-1,1,0,1,1,0,0,1,1,0,1,1,0,-1]
    
    x1 =    [[1,0,0, 1,0,0, 1,1,1, -1], [0,1,0, 0,1,0, 0,1,0, -1],
             [1,0,0, 1,0,0, 1,0,0, -1], [0,0,1, 0,0,1, 0,0,1, -1], 
             [0,1,0, 0,1,0, 0,1,1, -1]]

    x2 =    [[1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, -1],
             [1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1, -1],
             [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, -1],
             [0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1, -1]]

    x3 =    [[1,1,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1, -1],
             [1,1,1,0,0, 1,0,0,0,0, 1,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0, -1],
             [1,1,1,1,1, 1,1,1,1,1, 1,1,0,0,0, 1,1,1,1,1, 1,1,1,1,1, -1],
             [1,0,1,0,0, 1,0,1,0,0, 1,1,1,0,0, 0,0,0,0,0, 0,0,0,0,0, -1],
             [1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0, -1],
             [1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 1,1,1,1,1, -1]]
             
    t1 = [1,-1,-1,-1,1]
    t2 = [-1,1,-1,1]
    t3 = [1,1,1,-1,-1,-1]
    neuron1 = Neuron(w_initial1, 0.4)
    neuron2 = Neuron(w_initial2, 0.4)
    neuron3 = Neuron(w_initial2, 0.4)
    
    print "Starting training for 3x3 ..."
    neuron1.trainNeuron(x1,t1)
    print "Finished training for 3x3."

    print "Starting training for 5x5 ..."
    neuron2.trainNeuron(x2,t2)
    neuron3.trainNeuron(x3,t3)
    print "Finished training for 5x5."
    
    printResults(x1,t1,neuron1)
    printResults(x2,t2,neuron2)
    printResults(x3,t3,neuron3)
    
def printResults(x,t,neuron):
    file = open("results", "a")
    
    for i in range(len(x)):
        y = neuron.evaluate(x[i])
        file.write("Neuron evaulated to ")
        file.write(str(y))
        file.write(" expected ")
        file.write(str(t[i]))
        file.write("\n")

    file.write("Final weights of neuron:\n")
    file.write(" ".join('%0.2f' % item for item in neuron.w))
    file.write("\n\n")
    
    
class Neuron:
    w = []
    LEARN_RATE = -1

    def __init__(self, w_initial, learnRate):
        self.LEARN_RATE = learnRate
        self.w = list(w_initial)
    
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
        
        while not successful:
            successful = True

            order = random.sample(range(len(x)), len(x))

            for i in order:
                y = self.evaluate(x[i])

                if (y == t[i]):
                    continue
                    
                else:
                    successful = False
                    self.adjustWeights(x[i],y,t[i])
        
    def adjustWeights(self, x, y, t):
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.LEARN_RATE * (y-t) * x[i]
                        
main()
