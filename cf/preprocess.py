from code import interact
from _operator import itemgetter
from test.test_decimal import Coverage
def readData(file):
    data = []
    fin = open(file, 'r')
    for line in fin:
        user, item = line.split('::')[0:2]
        user = int(user)
        item = int(item)
        data.append([user, item])
    fin.close()
    return data
# end read data

import random
import math
# #data=readData('data.txt'); for k,v in data: print(k,v)
def SplitData(data, k=0, seed=1, M=10):
    test = []
    train = []
    random.seed(seed)
    for user , item in data:
        if random.randint(0, M) == k:
            test.append([user, item])
        else:
            train.append([user, item])

    return train, test

# end splitdata
# 返回的map，key为userid， 值为item id的set
def getMap(dataL):
    retM = dict()
    for user, item in dataL:
        if user not in retM:
            retM[user] = set() 
        retM[user].add(item)
    return retM


# print(testM)
# #print(len(test),len(train))

# #for k,v in testM.items(): print(k,v)

def Recall(train, test, ranks):
    hit = 0
    all = 0
    
    for user in train.keys():
#         count=0
        if user not in test.keys():
            continue
        tu = test[user]
        rank = ranks[user]
#         rank=Recommend(user, train, W)
        for item, score in rank:
            if item in tu:
                hit += 1
#                 count=count+1
        all += len(tu)
#         print(count/len(tu))
    return hit / (all * 1.0)
# end recall

def Precision(train, test, ranks):
    hit = 0
    all = 0
    for user in train.keys():
        if user not in test.keys():
            continue
        tu = test[user]
        rank = ranks[user]
        for item, score in rank:
            if item in tu:
                hit += 1
        all += len(rank)
    return hit / (all * 1.0)
# end precision    
def Coverage(train, test, ranks):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank = ranks[user]
        for item, score in rank:
            recommend_items.add(item)
    return len(recommend_items) / len(all_items)
# end coverage
def Popularity(train, test, ranks):
    item_popularity = dict()
    for user, items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    
    ret = 0
    n = 0
    for user in train.keys():
        rank = ranks[user]
        for item, score in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    
    ret /= n
    return ret
# end popularity




    
    
            


                        
