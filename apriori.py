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
            H1 = [frozenset([item]) for item in freqSet]
            print("{}. freqSet:{}, H1:{}, L[i]:{}".format(i, freqSet, H1, L[i]))
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet - conseq]
        if conf >= minConf:
            print("freqSet-conseq:{} ---> {}, conf: {}".format(freqSet-conseq, conseq, conf))
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
        else:
            print("freqSet-conseq:{} ---> {}, conf: {} <= minConf, ignore".format(freqSet-conseq, conseq, conf))
    return prunedH


def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    Hmp1 = calcConf(freqSet, H, supportData, br1, minConf)
    print("H-length:{}, freqSet:{}".format(m, len(freqSet)))
    if (len(freqSet) > (m + 1)):
        print(H)
        print(Hmp1)
        Hmp1 = aprioriGen(Hmp1, m+1)
        print(Hmp1)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


file_path_pattern = "/home/lty/machine-learn-space/scripts/python-workspace/machinelearninginaction/Ch11/{}"
mushroom_path = file_path_pattern.format("mushroom.dat")

mushDatSet = [line.split() for line in open(mushroom_path).readlines()]
for mushDat in mushDatSet:
    print(mushDat)

L, suppData = apriori(mushDatSet, minSupport=0.3)

print("##############################")
for i in range(len(L)):
    if len(L[i]) > 0:
        print("{}.{}".format(len(L[i][0]), L[i]))
print("##############################")
# print(suppData)

# dataSet = loadDataSet()
# print(dataSet)
# C1 = createC1(dataSet)
# # print(list(C1))
# D = dataSet
# print(list(D))

# L1, suppData0 = scanD(D, C1, 0.5)
# print(L1)
# print(suppData0)

# L, suppData = apriori(dataSet, minSupport= 0.5)
# print(L)
# for lItem in L:
#     print(lItem)


# br1 = []

# for i in range(1, len(L)):
#     print("{}: {}".format(i, L[i]))
#     for freqSet in L[i]:
#         prunedH = []
#         H = [frozenset([item]) for item in freqSet]
#         print("freqSet:{}".format(freqSet))
#         print("freqSet-length:{}".format(len(freqSet)))
#         for conseq in H:
#             print("conseq:{}".format(conseq))
#             print("conseq-length:{}".format(len(conseq)))
#             print("freqSet-conseq:{}".format(freqSet - conseq))
#             # print(suppData[freqSet])
#             # print(suppData[freqSet - conseq])
#             conf = suppData[freqSet]/suppData[freqSet - conseq]
#             br1.append((freqSet - conseq, conseq, conf))
#             prunedH.append(conseq)
#         print(prunedH)
#         print("###############")
#     # H = [frozenset([item]) for item in L[i]]

# print(br1)


# L, suppData = apriori(dataSet, minSupport=0.2)
# print(L)
# rules = generateRules(L, suppData, minConf=0.7)
# print("#################")
# print(rules)

