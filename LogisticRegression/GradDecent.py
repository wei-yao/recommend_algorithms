from math import exp
from numpy import *


def loadData(file):
    labelMat = []
    dataMat=[]
    with open(file, 'r') as fin:
        for line in fin.readlines():
            lineAttr = line.strip().split()
            # why 1.0 大概为了方便画图
            row=[1.0]
            row.extend([float(attr) for attr in lineAttr[:-1]])
            dataMat.append(row)
            labelMat.append(int(lineAttr[-1]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))
    # return inX


def gradDecent(dataMatrixIn, labels):
    dataMatrix = mat(dataMatrixIn)
    # 直接转换会转换成一个行向量
    labelMat = mat(labels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    round = 500
    weights = ones((n, 1))
    for i in range(round):
        predictedLabels = sigmoid(dataMatrix * weights)
        error = predictedLabels - labelMat
        weights -= alpha * dataMatrix.transpose() * error
    return weights


def randomGradDecent(dataMatrixIn, labels, round=500):
    m = len(dataMatrixIn)
    n = len(dataMatrixIn[0])
    weights = ones(n)
    for j in range(round):
        perm = random.permutation(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            index = perm[i]
            predictLabel = sigmoid(sum(weights * dataMatrixIn[index]))
            error = predictLabel - labels[index]
            weights -= alpha * error * dataMatrixIn[index]
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadData('testSet.txt')
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()





def loadHorseData(file):
    labelMat = []
    dataMat=[]
    with open(file, 'r') as fin:
        for line in fin.readlines():
            lineAttr = line.strip().split()
            # why 1.0 大概为了方便画图
            dataMat.append([float(attr) for attr in lineAttr[:-1]])
            labelMat.append(float(lineAttr[-1]))
    return dataMat, labelMat


def predict(attr,weights):
    if sigmoid(sum(attr*weights))>0.5:
        return 1.0
    else:
        return 0



def colicTest():
    trainFile = 'horseColicTraining.txt'
    testFile = 'horseColicTest.txt'
    trainData,trainLabels=loadHorseData(trainFile)
    testData,testLabels=loadHorseData(testFile)
    weights=randomGradDecent(array(trainData),trainLabels)
    errorCount=0
    for i in range(len(testData)):
        if predict(testData[i],weights)!=testLabels[i]:
            errorCount+=1

    return errorCount/len(testLabels)