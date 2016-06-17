'''
Created on 2016年4月16日
user cf
@author: Administrator
'''
import math
from _operator import itemgetter
from root.nested.preprocess import readData, SplitData, getMap, Recall,\
    Precision, Coverage, Popularity

def ItemSimilarity(train):
    C=dict()
    N=dict()
    for u,items in train.items():
        for i in items:
            if i not in N:
                N[i]=0
            if i not in C:
                C[i]=dict()
            N[i]+=1
            for j in items:
                if i==j:
                    continue
                if j not in C[i]:
                    C[i][j]=0
                C[i][j]+=1/math.log(1+len(items))
#                 C[i][j]+=1
    W=dict()
    maxI=dict();
    for i ,related_items in C.items():
        if i not in W:
            W[i]=dict()
        if i not in maxI:
            maxI[i]=0
        for j,cij in related_items.items():
            W[i][j]=cij/math.sqrt(N[i]*N[j])
            if(W[i][j]>maxI[i]):
                maxI[i]=W[i][j]
        for j in related_items.keys():
            W[i][j]=W[i][j]/maxI[i]
    return W
#end item similarity

def Recommendation(train,user_id,W,K=10):
    rank=dict()
    ru=train[user_id]
    for i in ru:
        for j,wj in sorted(W[i].items(),key=itemgetter(1),reverse=True)[0:K]:
            if j in ru:
                continue
            if j not in rank:
                rank[j]=0
            rank[j]+=wj
    return rank.items()
    end=10
    if len(rank) < end:
        end = len(rank)
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:end]
#end recomendation
smallData = "small.txt";
bigData = "ratings.dat";
            
data = readData(bigData);
recall=0
precision=0
coverage=0
popularity=0
roundNum=1
for i in list(range(0, roundNum)):
    trainL , testL = SplitData(data, i, 1, roundNum)
    train = getMap(trainL)
    test = getMap(testL)
    W = ItemSimilarity(train)
    #     W = UserSimilarityNormal(train)
    ranks = dict();
    for user in train.keys():
        ranks[user] = Recommendation(train,user, W)
        
    recall += Recall(train, test, ranks)
    precision += Precision(train, test, ranks)
    print("roundNum ", i)
    coverage+=Coverage(train, test, ranks)
    popularity+=Popularity(train, test, ranks)
# #print(w)
print("recall: ", recall/roundNum)
print('precision', precision/roundNum)
print('coverage', coverage/roundNum)
print('popularity', popularity/roundNum)




