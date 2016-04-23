#惩罚热门物品的相似度
import math
from _operator import itemgetter
from root.nested.preprocess import readData, SplitData, getMap, Recall,\
    Precision, Coverage, Popularity
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
#     return rank.items()
    if len(rank) < end:
        end = len(rank)
    return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:end]
# end def recommend
smallData = "small.txt";
bigData = "ratings.dat";
          
data = readData(smallData);
recall=0
precision=0
coverage=0
popularity=0
roundNum=1
print("usercf")
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