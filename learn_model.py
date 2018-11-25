# -*- coding: utf-8 -*-
"""
@author: Valeri
"""

import matplotlib.pyplot as pl
import time
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from sklearn.cross_validation import train_test_split
from pybrain.supervised.trainers import BackpropTrainer
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold

# Default values:
n_in = 15
n_hidden0 = 15
n_out = 3
datafile = 'data/data_procc.csv'
classes = ["rock", "paper", "scissors", "unknown"]
momentum = 0.3
learningrate = 0.0001
epochs = 240


def convertToDataSet(X, Y):
    ds = SupervisedDataSet(n_in, n_out)
    for i in range(len(X)):
        tempArr = [0, 0, 0]
        tempArr[int(Y[i])] = 1
        ds.addSample(X[i], tempArr)
    return ds


def getClasses(output):
    arrClasses = []
    for out in output:
        np_out = np.asarray(out)      
        max_index = np_out.argmax()  
        arrClasses.append(max_index)
        # TODO: in other use cases, might want to say "unknown"
    return arrClasses


def plotPrecVsEpoch(trainDS, testDS, figTitle):
    net = buildNetwork(n_in, n_hidden0, n_out)
    print(net['in'])
    print(net['hidden0'])
    print(net['out'])

    trainer = BackpropTrainer(net, trainDS, momentum=momentum, learningrate=learningrate, verbose=True)

    test_prec = []
    train_prec = []

    for i in range(epochs):
        trainer.train()
        # Test precision
        out = net.activateOnDataset(trainDS)
        np_out = getClasses(out)
        train_out = getClasses(trainDS['target'])
        train_prec.append( metrics.precision_score(np_out, train_out, average='macro') )
        
        # Train precision
        out = net.activateOnDataset(testDS)
        np_out = getClasses(out)
        test_out = getClasses(testDS['target'])
        test_prec.append(metrics.precision_score(np_out, test_out, average='macro') )
    
    # TODO: print full-blown metrics here?
    #print 'train_prec'
    #print train_prec
    #print 'test_prec'
    #print test_prec
    
    print(n_hidden0, ", ", learningrate, ", ", max(train_prec), ", ", train_prec[-1], ", ", max(test_prec), ", ", test_prec[-1])

    pl.ioff()
    pl.plot(train_prec, color='blue', label='Precision on training set')
    pl.plot(test_prec, color='red', label='Precision on testing set')
    pl.legend(loc=4)
    pl.savefig(figTitle, bbox_inches='tight')
    pl.clf()
    

def run(trainDS, testDS, m):
    net = buildNetwork(n_in, n_hidden0, n_out) 
    trainer = BackpropTrainer(net, trainDS, momentum=m) 
    
    trnerr,valerr = trainer.trainUntilConvergence(trainDS,maxEpochs=30)
    
    out = net.activateOnDataset(testDS)
    # round to nearest int, so it matches testDS formatting
    np_out = np.asarray(out)
    # TODO: handle classes the right way
    indeces_1 = out >= 0.5
    indeces_0 = out < 0.5
    np_out[indeces_1] = 1
    np_out[indeces_0] = 0
    precision = metrics.precision_score(np_out, testDS['target'])

    return net, precision


def runCV(trainX, trainY, m):
    scores = []
    for train_index, test_index in cv:
        trainDS_train = convertToDataSet(trainX[train_index], trainY[train_index])
        trainDS_val = convertToDataSet(trainX[test_index], trainY[test_index])
        temp_net, temp_score = run(trainDS_train, trainDS_val, m)
        scores.append(temp_score)
    avg = np.average(scores)
    print('For m = ' + str(m) + ' avg precision was: ' + str(avg))
    return avg


def runExploratoryRuns():
    dataset = np.loadtxt(datafile, delimiter=',')

    print("Dataset shape: ", dataset.shape)

    X = dataset[:, 1:16]
    Y = dataset[:, 0]
    trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.65, test_size=0.35, random_state=42)

    trainDS = convertToDataSet(trainX, trainY)
    testDS = convertToDataSet(testX, testY)

    # TODO: Are these the right arguments for a stratified kFold?
    # cv = StratifiedKFold(trainY, 5)

    mRange = [0, 0.2, 0.4, 0.6, 0.8]
    hiddenRange = [5, 10, 15, 20, 25, 30]
    learningRange = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    learningrate = 0.00001
    n_hidden0 = 25
    epochs = 100
    plotPrecVsEpoch(trainDS, testDS, "figures/Test title.png")


start = time.clock()



end = time.clock()

print("Time elapsed: ")
print (end - start)