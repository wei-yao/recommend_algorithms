##classification tree algorithm
from operator import *

import sys

from math import log


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels


def majorityCnt(dataSet):
    vote={}
    for row in dataSet:
        classLable=row[-1]
        if classLable not in vote:
            vote[classLable]=0
        vote[classLable]+=1
    return sorted(vote.items(),key=itemgetter(1),reversed=True)[0][0]


def calEntropy(data):
    classDic={}
    for row in data:
        classDic[row[-1]]=classDic.get(row[-1],0)+1
    entropy=0
    for k,v in classDic.items():
        prob=v/len(data)
        entropy+=-log(prob,2)*prob
    return entropy


def chooseBestSplit(dataSet):
    featureLen=len(dataSet[0])-1
    minValue=sys.maxsize
    minIndex=-1
    for i in range(featureLen):
        splitResult={}
        for row in dataSet:
            if row[i] not in splitResult:
                splitResult[row[i]]=[]
            splitResult[row[i]].append(row)
        averageEntropy=0
        for k,v in splitResult.items():
            averageEntropy+=calEntropy(v)*len(v)/len(dataSet)
        if averageEntropy<minValue:
            minValue=averageEntropy
            minIndex=i
    return minIndex
def splitByAttr(dataSet, splitIndex, value):
    dataAfterSplit=[]
    for row in dataSet:
        if row[splitIndex]==value:
            temp=row[0:splitIndex]
            temp.extend(row[splitIndex+1:])
            dataAfterSplit.append(temp)
    return dataAfterSplit


def createTree(dataSet,dataLabels):
    #copy
    labels=dataLabels[:]
    classList=set([row[-1] for row in dataSet])
    #叶子节点的值是标签
    if len(classList)==1:
        return dataSet[0][-1]
    if len(dataSet[0])==1:
        return majorityCnt(dataSet)
    bestSplit=chooseBestSplit(dataSet)
    bestSplitLabel=labels[bestSplit]
    node={bestSplitLabel:{}}
    del(labels[bestSplit])
    splitValues=set([row[bestSplit] for row in dataSet])
    for value in splitValues:
        node[bestSplitLabel][value]=createTree(splitByAttr(dataSet,bestSplit,value),labels)
    return node

def classify(inputTree,featLabels,inputVect):
    splitFeature=list(inputTree.keys())[0]
    secondDict=inputTree[splitFeature]
    splitIndex=featLabels.index(splitFeature)
    for k in secondDict.keys():
        if inputVect[splitIndex]==k:
            if type(secondDict[k]) is dict:

                classLabel=classify(secondDict[k],featLabels,inputVect)
            else:
                classLabel=secondDict[k]
    return classLabel