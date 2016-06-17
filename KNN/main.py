from numpy.ma import array

from KNN import *
import matplotlib
import matplotlib.pyplot as plt

from KNN.Knn import *

matrix, label = file2matrix('datingTestSet2.txt')
testKnn(matrix,label)
# print(matrix)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(matrix[:, 0], matrix[:, 1],15.0*array(label),15.0*array(label))
# plt.show()
# a=15.0*array(label)
# print(a)
