import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    # returns the value used for each leaf
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].flatten().A[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # print("feat:{}, val:{}".format(feat, val))
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("Merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0:
        raise NameError('This Martix is singular, cannot do inverse, \ntry increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = inDat
    return float(X*model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


import tkinter
import matplotlib
matplotlib.use('Tkagg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = createTree(reDraw.rawDat, modelLeaf, modelErr, (tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat, modelTreeEval)
    else:
        myTree = createTree(reDraw.rawDat, regLeaf, regErr, (tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat, regTreeEval)
    reDraw.a.scatter(reDraw.rawDat[:, 0].flatten().A[0], reDraw.rawDat[:, 1].flatten().A[0], s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0, color="brown")
    reDraw.canvas.draw()


def getInputs():
    try:
        tolN = int(tolNEntry.get())
    except:
        tolN = 10
        print("Enter Integer for tolN")
        tolNEntry.delete(0, np.END)
        tolNEntry.insert(0, '10')
    try:
        tolS = float(tolSEntry.get())
    except:
        tolS = 1.0
        print("Enter Float for tolS")
        tolSEntry.delete(0, np.END)
        tolSEntry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)




root = tkinter.Tk()
# row 0
# tkinter.Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
# row 1
tkinter.Label(root, text="tolN").grid(row=1, column=0)
tolNEntry = tkinter.Entry(root)
tolNEntry.grid(row=1, column=1)
tolNEntry.insert(0, '10')
# row 2
tkinter.Label(root, text='tolS').grid(row=2, column=0)
tolSEntry = tkinter.Entry(root)
tolSEntry.grid(row=2, column=1)
tolSEntry.insert(0, '1.0')
tkinter.Button(root, text='reDraw', command=drawNewTree).grid(row=2, column=2, rowspan=2)
# row 3
chkBtnVar = tkinter.IntVar()
chkBtn = tkinter.Checkbutton(root, text='Model Tree', variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)
# row 4
# tkinter.Button(root, text='Quit', fg='black', command=root.quit).grid(row=3, column=2)

file_path_pattern = "/home/lty/machine-learn-space/scripts/python-workspace/machinelearninginaction/Ch09/{}"
sine_path = file_path_pattern.format("sine.txt")

reDraw.rawDat = np.mat(loadDataSet(sine_path))
reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)



root.mainloop()







testMat = np.mat(np.eye(4))
print(testMat[:, -1].flatten().A[0])
# print(np.shape(testMat))
# print(testMat[:, -1].T.tolist()[0])
print(set(testMat[:, 1].flatten().A[0]))
# print(regLeaf(testMat))
# print(np.nonzero(testMat[:, 0] >= 1)[0])
# print(testMat[np.nonzero(testMat[:, 0] >= 1)[0], :])

file_path_pattern = "/home/lty/machine-learn-space/scripts/python-workspace/machinelearninginaction/Ch09/{}"
ex00_path = file_path_pattern.format("ex00.txt")
ex0_path = file_path_pattern.format("ex0.txt")
ex2_path = file_path_pattern.format("ex2.txt")
ex2test_path = file_path_pattern.format("ex2test.txt")
exp2_path = file_path_pattern.format("exp2.txt")
bikeSpeedTrain_path = file_path_pattern.format("bikeSpeedVsIq_train.txt")
bikeSpeedTest_path = file_path_pattern.format("bikeSpeedVsIq_test.txt")

# myData = loadDataSet(exp2_path)
# # print(myData)
# myMat = np.mat(myData)
# myTree = createTree(myMat, modelLeaf, modelErr, (1, 10))
# print(myTree)

# print("###########################################")
# myTestData = loadDataSet(ex2test_path)
# myTestMat = np.mat(myTestData)
# myTree = prune(myTree, myTestMat)
# print(myTree)

trainMat = np.mat(loadDataSet(bikeSpeedTrain_path))
testMat = np.mat(loadDataSet(bikeSpeedTest_path))
myTree = createTree(trainMat, ops=(1, 20))
yHat = createForeCast(myTree, testMat[:, 0])
print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])


myTree = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

