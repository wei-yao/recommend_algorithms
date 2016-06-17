import re

from os import listdir
import random

from numpy.ma import array, ones, log


def text2wordList(file):
    with open(file, 'r', encoding='utf-8', errors='ignore') as fin:
        text = fin.read()
    return textParse(text)


def textParse(text):
    tokenList = re.split(r'\W*', text)
    return [tok.lower() for tok in tokenList if len(tok) > 2]


def loadDocList(dir):
    fileList = listdir(dir)
    docList = []
    for file in fileList:
        docList.append(text2wordList(dir + '/' + file))
    return docList


def word2vec(wordList, wordIndexDic):
    wordVec = [0] * len(wordIndexDic)
    for w in wordList :
        if w in wordIndexDic:
            wordVec[wordIndexDic[w]] += 1
    return wordVec


def trainNB0(trainMatrix, trainLabels):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainLabels[i] == 1:
            # 这里计算的p(w|c) 用的是类别中word w 出现的总数 除以类别 c 中词的总数，也就是将类别中所有的词作为一个大词袋
            # 当然也可以以 文档为单位计算，那么p(w=i|c)= （c中w=i的文档数)/（c 中文档总数)
            p1Denom += sum(trainMatrix[i])
            p1Num += trainMatrix[i]
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p0Vec = log(p0Num / p0Denom)
    p1Vec = log(p1Num / p1Denom)
    pAbusive = sum(trainLabels) / len(trainLabels)
    return p0Vec, p1Vec, pAbusive


def predictNBO(p0Vec, p1Vec, pAbusive, inputVec):
    p1 = sum(inputVec * p1Vec) + log(pAbusive)
    p0 = sum(inputVec * p0Vec) + log(1.0 - pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0


def spamTest():
    spamDocList = loadDocList('email/spam')
    hamDocList = loadDocList('email/ham')
    docList = []
    classList = []
    docList.extend(spamDocList)
    docList.extend(hamDocList)
    classList.extend([1 for i in range(len(spamDocList))])
    classList.extend([0 for i in range(len(hamDocList))])
    wordSet = set()
    for doc in docList:
        for word in doc:
            wordSet.add(word)
    wordIndexDic = {}
    index = 0
    for w in wordSet:
        wordIndexDic[w] = index
        index += 1

    totalDocSize = len(docList)
    testSize = int(totalDocSize * 0.2)
    testIndex = set(random.sample(range(totalDocSize), testSize))
    trainingSet = []
    testSet = []
    trainingSet = []
    trainLabel = []
    for i in range(totalDocSize):
        if i not in testIndex:
            trainingSet.append(word2vec(docList[i], wordIndexDic))
            trainLabel.append(classList[i])
    p0V, p1V, pSpam = trainNB0(array(trainingSet), array(trainLabel))
    errorNum = 0.0
    for i in testIndex:
        predictedLabel = predictNBO(p0V, p1V, pSpam, word2vec(docList[i], wordIndexDic))
        if predictedLabel != classList[i]:
            errorNum += 1
    return errorNum / len(testIndex)
