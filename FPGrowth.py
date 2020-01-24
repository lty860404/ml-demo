

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=2):
        print('{}{}: {}'.format(" "*ind, self.name, self.count))
        for child in self.children.values():
            child.disp(ind + 4)


def createTree(dataSet, minSup=1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePath, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, prefix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    print("bigL:{}".format(bigL))
    for basePat in bigL:
        newFreqSet = prefix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print("base Pat:{}, condPattBases:{}".format(basePat, condPattBases))
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead is not None:
            print("conditional tree for: {}, freqItemList:{}".format(newFreqSet, freqItemList))
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
        else:
            print("Header is None for {} freqItemList:{}".format(newFreqSet, freqItemList))
        print("------------------------------------")


simpDat = loadSimpDat()
# print(simpDat)
# print("########################################")
initSet = createInitSet(simpDat)
# print(initSet)

myFPTree, myHeaderTab = createTree(initSet, 3)
print(list(myHeaderTab.items()))
for item in list(myHeaderTab.items()):
    print(item[1][1])
myFPTree.disp()

print("########################################")


# xPrefixPath = findPrefixPath('x', myHeaderTab['x'][1])
# print(xPrefixPath)

# zPrefixPath = findPrefixPath('z', myHeaderTab['z'][1])
# print(zPrefixPath)

# rPrefixPath = findPrefixPath('r', myHeaderTab['r'][1])
# print(rPrefixPath)

# rootNode = treeNode('pyramid', 9, None)
# rootNode.children['eye'] = treeNode('eye', 13, None)
# rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
# rootNode.disp()

freqItems = []
mineTree(myFPTree, myHeaderTab, 3, set([]), freqItems)
print(freqItems)


file_path_pattern = "/home/lty/machine-learn-space/scripts/python-workspace/machinelearninginaction/Ch12/{}"
kosarak_path = file_path_pattern.format("kosarak.dat")

parsedDat = [line.split() for line in open(kosarak_path).readlines()]
kosarakInitSet = createInitSet(parsedDat)
kosarakTree, kosarakHeaderTab = createTree(kosarakInitSet, 100000)
kosarakList = []
mineTree(kosarakTree, kosarakHeaderTab, 100000, set([]), kosarakList)

print("################################")
print("frequent kosarak Item:")
for kosarakItem in kosarakList:
    print(kosarakItem)


