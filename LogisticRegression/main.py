from LogisticRegression.GradDecent import *


def main():
    dataMat, labelMat = loadData('testSet.txt')
    # weights = gradDecent(dataMat, labelMat)
    weights = randomGradDecent(array(dataMat), labelMat)
    plotBestFit(weights)


# main()
for i in range(10):
    print(colicTest())