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
#惩罚热门物品的相似度
def UserSimilarity(train):
    item_users = dict()
    for u, items in train.items():
        for i in items:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)


    C = dict();
    N = dict();
    for i, users in item_users.items():
        for u in users:
            if u not in C:
                C[u] = dict()
            if u not in N:
                N[u] = 1
            else:
                N[u] += 1;
            
            for v in users:
                if u == v:
                    continue
                if v not in C[u]:
                    C[u][v] = 0
              
                C[u][v] += 1/math.log(1+len(users))


    W = dict()
    for u , related_users in C.items():
        if u not in W:
            W[u] = dict()
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])

    return W
#余弦距离
def UserSimilarityNormal(train):
    item_users = dict()
    for u, items in train.items():
        for i in items:
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)


    C = dict();
    N = dict();
    for i, users in item_users.items():
        for u in users:
            if u not in C:
                C[u] = dict()
            if u not in N:
                N[u] = 1
            else:
                N[u] += 1;
            
            for v in users:
                if u == v:
                    continue
                if v not in C[u]:
                    C[u][v] = 0
              
                C[u][v] += 1


    W = dict()
    for u , related_users in C.items():
        if u not in W:
            W[u] = dict()
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])

    return W
def Recommend(user, train, W, K=5):
    rank = dict()
    interact_items = train[user]
    for v, wuv in sorted(W[user].items(), key=itemgetter(1), reverse=True)[0:K]:
        for item in train[v]:
            if item in interact_items:
                continue
            if item not in rank:
                rank[item] = 0
            rank[item] += wuv * 1
#     return rank
    end = 10
    if len(rank) < end:
        end = len(rank)
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:end]
# end def recommend
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

smallData = "small.txt";
bigData = "ratings.dat";
        
data = readData(smallData);
recall=0
precision=0
coverage=0
popularity=0
roundNum=10
for i in list(range(0, roundNum)):
    trainL , testL = SplitData(data, i, 1, roundNum)
    train = getMap(trainL)
    test = getMap(testL)
#     W = UserSimilarity(train)
    W = UserSimilarityNormal(train)
    ranks = dict();
    for user in train.keys():
        ranks[user] = Recommend(user, train, W)
        
    recall += Recall(train, test, ranks)
    precision += Precision(train, test, ranks)
#     print("roundNum ", i)
    coverage+=Coverage(train, test, ranks)
    popularity+=Popularity(train, test, ranks)
# #print(w)
print("recall: ", recall/roundNum)
print('precision', precision/roundNum)
print('coverage', coverage/roundNum)
print('popularity', popularity/roundNum)


    
    
            


                        
