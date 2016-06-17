from numpy import *
import  operator

def file2matrix(fileName):
    matrix = []
    labels = []
    with open(fileName, 'r') as fin:
        for line in fin.readlines():
            params = line.strip().split('\t')
            matrix.append(params[0:3])
            labels.append(int(params[3]))
    dataMatrix = zeros((len(matrix), 3))
    index = 0
    for params in matrix:
        dataMatrix[index, :] = params
        index += 1
    return autoNorm(dataMatrix), labels


def autoNorm(dataSet):
    minV = dataSet.min(0)
    maxV = dataSet.max(0)
    ranges = maxV - minV
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minV, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet


def classifyKnn(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    diffMat **= 2
    sqDistances = diffMat.sum(axis=1)**0.5
    sortedDistance = sqDistances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistance[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def testKnn(dataSet, labels):
    ratio = 0.1;
    testNum = int(dataSet.shape[0] * ratio)
    dataSize=int(dataSet.shape[0])
    k = 5
    errorCount = 0
    for i in range(testNum):
        label = classifyKnn(dataSet[i, :], dataSet[testNum:, :], labels[testNum:], k)
        if label != labels[i]:
            errorCount += 1
    print("error rate :",errorCount/testNum)