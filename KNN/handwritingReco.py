from numpy import *
from os import listdir
from KNN.Knn import *

DigitSize = 1024


def img2vector(filename):
    returnVect = zeros(1024)
    index = 0
    with open(filename, 'r') as fin:
        for line in fin.readlines():
            for i in range(32):
                returnVect[index] = int(line[i])
                index += 1
    return returnVect


def readData(dir):
    fileList = listdir(dir)
    data = zeros((len(fileList), DigitSize))
    labels = []
    i = 0
    for file in fileList:
        label = file.split('_')[0]
        labels.append(int(label))
        data[i, :] = img2vector(dir+'/'+file)
        i += 1
    return data, labels


def handwritingClassTest():
    trainDataDir = 'digits/trainingDigits'
    testDataDir = 'digits/testDigits'
    trainData, trainLabels = readData(trainDataDir)
    testData, testLabels = readData(testDataDir)
    trueCount = 0
    testDataSize=testData.shape[0]
    for i in range(testDataSize):
        data=testData[i,:]
        label=testLabels[i]

        predictedLabel = classifyKnn(data, trainData, trainLabels, 5)
        if predictedLabel == label:
            trueCount += 1
    accuracy = trueCount / len(testLabels)
    print(accuracy)

handwritingClassTest()