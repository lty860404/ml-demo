def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)


def scanD(D, Ck, minSupport):
    ssCnt = {}
    Cklist = [can for can in list(Ck)]
    for tid in D:
        # print(tid)
        for can in Cklist:
            # print(can)
            if can.issubset(tid):
                if not ssCnt.__contains__(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    # print(numItems)
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L1.sort()
            L2 = list(Lk[j])[:k-2]
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = dataSet.copy()
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item] for item in freqSet)]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    # prunedH = []
    # for conseq in H:
    return None




def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])

    return None





dataSet = loadDataSet()
print(dataSet)
C1 = createC1(dataSet)
# print(list(C1))
D = dataSet
# print(list(D))

# L1, suppData0 = scanD(D, C1, 0.5)
# print(L1)
# print(suppData0)

L, suppData = apriori(dataSet, minSupport= 0.5)
print(L)
for lItem in L:
    print(lItem)


br1 = []

for i in range(1, len(L)):
    print("{}: {}".format(i, L[i]))
    for freqSet in L[i]:
        prunedH = []
        H = [frozenset([item]) for item in freqSet]
        print("freqSet:{}".format(freqSet))
        print("freqSet-length:{}".format(len(freqSet)))
        for conseq in H:
            print("conseq:{}".format(conseq))
            print("conseq-length:{}".format(len(conseq)))
            print("freqSet-conseq:{}".format(freqSet - conseq))
            # print(suppData[freqSet])
            # print(suppData[freqSet - conseq])
            conf = suppData[freqSet]/suppData[freqSet - conseq]
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
        print(prunedH)
        print("###############")
    # H = [frozenset([item]) for item in L[i]]

print(br1)
