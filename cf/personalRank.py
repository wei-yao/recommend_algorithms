# G:graph, dict: key: user or item node, value: the adjacent node of key  root:the user , ie the start point
from root.nested.preprocess import readData, SplitData, getMap, Recall,\
    Precision, Coverage, Popularity
from _operator import itemgetter
def PersonalRank(G,root,alpha=0.8):
    rank=dict()
    rank={x:0 for x in G.keys()}
    rank[root]=1
    for k in range(0,20):
        tmp={x:0 for x in G.keys()}
        for i,ri in G.items():
            for j  in ri:
                tmp[j]+=0.6*rank[i]/len(ri)
                if j==root:
                    tmp[j]+=1-alpha
        rank=tmp;
#         print(rank)
    return rank
#end  personal rank
# 图中user 和 item 都是顶点，因此，需要对itemId 处理：itemId+=max(userId)
# 图实际就是将 user_items 表 和 item_users 表拼起来,这两个表中的itemId 都经过了处理
def genGraph(train):
    G=dict()
    item_users=dict()
    maxUserId=max(train.keys());
    for user,items in train.items():
        for item in items:
            newItemId=item+maxUserId
            if newItemId not in item_users:
                item_users[newItemId]=set()
            item_users[newItemId].add(user)
    for user ,items in train.items():
        G[user]={x+maxUserId for x in items}
    for item, users in item_users.items():
        G[item]=users
    return G
#end def genGraph
def Recommend(graph,maxUserId,user, k=10):
#     maxUserId=max(train.keys())
    ret=dict()
    rank=PersonalRank(graph, user)
    for id, value in rank.items():
        if id>maxUserId :
#             this is important
            if id not in graph[user]:
                ret[id-maxUserId]=value
    return sorted(ret.items(),key=itemgetter(1),reverse=True)[0:k]

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
    ranks=dict()
    graph=genGraph(train)
    maxUserId=max(train.keys())
    for user in train.keys():
        ranks[user]=Recommend(graph, maxUserId, user)
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