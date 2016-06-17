import operator
import random

import feedparser
from numpy.ma import array

from NativeBayes.nativeBayes import textParse, word2vec, trainNB0, predictNBO


def calWordFreq(docList):
    wordFreq = {}
    for doc in docList:
        for word in doc:
            if word not in wordFreq:
                wordFreq[word] = 0
            wordFreq[word] += 1
    return wordFreq


def localWords(feedAddr1, feedAddr2):
    feed1 = feedparser.parse(feedAddr1)
    feed2 = feedparser.parse(feedAddr2)
    entries1 = feed1['entries']
    entries2 = feed2['entries']
    minLen = min(len(entries1), len(entries2))
    docList = []
    classList = []
    for i in range(minLen):
        docList.append(textParse(entries1[i]['summary']))
        classList.append(1)
        docList.append(textParse(entries2[i]['summary']))
        classList.append(0)
    wordFreq = calWordFreq(docList)
    wordSet=set(wordFreq.keys())
    sortedWordFreq=sorted(wordFreq.items(),key=operator.itemgetter(1),reverse=True)
    top30Words=[word[0] for word in sortedWordFreq[:30]]
    # for w in top30Words:
    #     if w in wordSet:
    #         wordSet.remove(w)
    wordIndexDic = {}
    index = 0
    for w in wordSet:
        wordIndexDic[w] = index
        index += 1
    totalDocSize = len(docList)
    testSize = int(totalDocSize * 0.1)
    print('test size',testSize)
    testIndex = set(random.sample(range(totalDocSize), testSize))
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
