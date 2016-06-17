from ClassificationTree import treePlotter
from ClassificationTree.CT import *

# dataSet,labels=createDataSet()
# print(createTree(dataSet,labels))
# treePlotter.createPlot(createTree(dataSet,labels))
# treePlotter.createPlot(treePlotter.retrieveTree(0))
# tree=createTree(dataSet,labels)
dataSet = [];
with open('lenses.txt', 'r') as fin:
    lenses = [line.strip().split('\t') for line in fin.readlines()]
labels=['age','prescript','astigmatic','tearRate']
tree=createTree(lenses,labels)
treePlotter.createPlot(tree)

# predictedLabel=classify(tree,labels,dataSet[0][:-1])
# print(dataSet[0])
# print(predictedLabel)
