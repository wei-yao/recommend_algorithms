#on MovieLens 10M dataset  content based knn， content  is extracted from the categories of the movie
from _operator import itemgetter

from root.nested.preprocess import readData, SplitData, getMap, Recall, \
    Precision, Coverage, Popularity
import math


def contentSimilarity(train,attrFile='movies.dat'):
    itemsSet=set()
    movies=dict()
    for items in train.values():
        itemsSet.update(items)
    with open(attrFile,'r',encoding='utf8') as handle:
        for line in handle:
            itemId,name,genre = line.split('::')[0:3]
            itemId=int(itemId)
            if itemId in itemsSet:
                genre=genre.rstrip()
                movies[itemId]=genre.split('|')
    keyItems=dict()
    documentFreq=dict()
    for itemId, contents in movies.items():
        for content in contents:
            if(content not in keyItems):
                keyItems[content]=set()
            keyItems[content].add(itemId)
            if content not in documentFreq:
                documentFreq[content]=0
            documentFreq[content]+=1
    W=dict()
    moviesV=dict()
    for key,items in keyItems.items():
        w=1/math.log(documentFreq[key])
        for i in items:
            if i not in W:
                W[i]=dict()
            if i not in moviesV:
                moviesV[i]=0
            moviesV[i]+=w*w
            for j in items:
                if(j==i):
                    continue
                if j not in W[i]:
                    W[i][j]=0
                W[i][j]+=w*w
    for i,relatedItems in W.items():
        for j in relatedItems:
            W[i][j]=W[i][j]/math.sqrt(moviesV[i]*moviesV[j])
    return W
#enddef contentSimilarity    
    
    
    
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
            
data = readData(smallData);
recall=0
precision=0
coverage=0
popularity=0
roundNum=1
print("contentKNN")
for i in list(range(0, roundNum)):
    trainL , testL = SplitData(data, i, 1, roundNum)
    train = getMap(trainL)
    test = getMap(testL)
    W = contentSimilarity(train)
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
    