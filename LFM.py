# -*- coding: utf-8 -*-  
from _operator import itemgetter
import random

from root.nested.preprocess import readData, SplitData, getMap, Recall, \
    Precision, Coverage, Popularity
import math
import pickle
# 物品池，允许有重复，这样，物品被随机抽中的概率与其热度成正比
items_pool = []
F = 100
N = 10
def RandomSelectNegativeSample(items):
    ret = dict()
    for i in items:
        ret[i] = 1
    n = 0
    for i in range(0, len(items) * 3):
        item = items_pool[random.randint(0, len(items_pool) - 1)]
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > len(items):
            break
    return ret
def InitModel(user_items, F):
    P = dict()
    Q = dict()
    for u in user_items.keys():
        if u not in P:
            P[u] = []
            for f in range(0, F):
# 这个是不是1/F 比较好
                P[u].append(random.random()*0.1)
    for items in user_items.values():
        for i in items:
            if i not in Q:
                Q[i] = []
                for f in range(0, F):
                    Q[i].append(random.random()*0.1)
    return [P, Q]
def Predict(user, item, P, Q):
    p = 0;
    for f in range(0, F):
        p += P[user][f] * Q[item][f]
    return p
    
def LatentFactorModel(user_items, alpha=0.02, lambda_=0.01):
    for items in user_items.values():
        for item in items:
            items_pool.append(item)
    [P, Q] = InitModel(user_items, F)
    last=999999
    for step in range(0, N):
        sum=0
        for user, items in user_items.items():
            samples = RandomSelectNegativeSample(items)
            
            for item, rui in samples.items():
                eui = rui - Predict(user, item, P, Q)
#                 print(eui)
                sum+=eui
                for f in range(0, F):
                    tempP= P[user][f]+ 2*alpha * (eui * Q[item][f] - lambda_ * P[user][f])
                    tempQ = Q[item][f]+ 2*alpha * (eui * P[user][f] - lambda_ * Q[item][f])
                    P[user][f]=tempP
                    Q[item][f]=tempQ
        print(sum)
        if(math.fabs(sum-last)<1 and math.fabs(sum)<100):
            break
        else:
            last=sum
        alpha *= 0.9       
# 当两次优化目标相差不大时，可以退出
        
    return [P, Q]
# end def latentfactormodel
def Recommend(user, P, Q,train, K=30):
    rank = dict()
    userItems=train[user];
    for item in Q.keys():
        if item in userItems:
            continue
        rank[item] = 0
        for f in range(0, F):
            rank[item] += P[user][f] * Q[item][f]  
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:K]
#end recommend

def classTopN(Q,data='movies.dat',output='topN.txt'):
    top =dict()
    movies=dict()
    k=10
    with open(data, 'r',encoding='utf8') as handle:
        for line in handle:
            itemId,name,genre = line.split('::')[0:3]
#             print(itemId)
            movies[int(itemId)]=name+'  '+genre
    for i in range(0,F):
        temp={}
        for item in Q.keys():
            temp[item]=Q[item][i]
        tempTop=sorted(temp.items(),key=itemgetter(1),reverse=True)[0:k]
        top[i]=tempTop
       
        with open(output,'a') as outputFile:
            outputFile.write(str(i)+'**********')
            for id,score in tempTop:
                outputFile.write(movies[id])
            outputFile.write('*****************')   
    return top
#end classTopN
smallData = "small.txt";
bigData = "ratings.dat";
          
data = readData(bigData);
recall = 0
precision = 0
coverage = 0
popularity = 0
roundNum = 1
print("LFM")
for i in list(range(0, roundNum)):
    trainL , testL = SplitData(data, i, 1, roundNum)
    train = getMap(trainL)
    test = getMap(testL)
# #     W = UserSimilarity(train)
# #     W = UserSimilarityNormal(train)
#     [P,Q]=LatentFactorModel(train, alpha=0.02, lambda_=0.01)  
#     with open('P.data', 'wb') as handle:
#         pickle.dump(P, handle)
#     with open('Q.data', 'wb') as handle:
#         pickle.dump(Q, handle)
    with open('P.data', 'rb') as handle:
        P=pickle.loads(handle.read())
    with open('Q.data', 'rb') as handle:
        Q=pickle.loads(handle.read())
    classTopN(Q)
#     ranks = dict();
#     for user in train.keys():
#         ranks[user] = Recommend(user, P, Q,train)
#    
#     
#     recall += Recall(train, test, ranks)
#     precision += Precision(train, test, ranks)
# #     print("roundNum ", i)
#     coverage += Coverage(train, test, ranks)
#     popularity += Popularity(train, test, ranks)
# # #print(w)
# print("recall: ", recall / roundNum)
# print('precision', precision / roundNum)
# print('coverage', coverage / roundNum)
# print('popularity', popularity / roundNum)

    
    
        
