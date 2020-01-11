import numpy as np
import matplotlib.pyplot as plt


data_base_dir = "/home/lty/machine-learn-space/scripts/python-workspace/machinelearninginaction/Ch08"
ext0_path = "{}/ex0.txt".format(data_base_dir)
ext1_path = "{}/ex1.txt".format(data_base_dir)
abalone_path = "{}/abalone.txt".format(data_base_dir)


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        lineInfos = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(lineInfos[i]))
        dataMat.append(lineArr)
        labelMat.append(float(lineInfos[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint*ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        if i % 10==0:
            print(i)
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


#yArr and yHatArr both need to be arrays
def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()


def ridgeRegress(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)   #calc mean then subtract it off
    inVar = np.var(inMat, 0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1.0, 1.0]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat





xArr, yArr = loadDataSet(abalone_path)
wsMat = stageWise(xArr, yArr, 0.01, 200)
# wsMat = stageWise(xArr, yArr, 0.001, 5000)
# [[ 0.044 -0.011  0.12   0.022  2.023 -0.963 -0.105  0.187]]
# print(wsMat)

xMat = np.mat(xArr)
yMat = np.mat(yArr).T
yMean = np.mean(yMat, 0)
yMat = yMat - yMean
xMat = regularize(xMat)

ws = standRegres(xMat, yMat.T)
print(ws.T)

# xArr = xArr[:99]
# yArr = yArr[:99]

# xMat = np.mat(xArr[:99])
# yMat = np.mat(yArr[:99])

# print(xArr)
# print(yArr)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# xCopy = xMat.copy()
# xCopy.sort(0)

# yHat = xCopy*ws

# print(np.shape(xArr[:99]))

# yHat = lwlrTest(xCopy, xArr[:99], yArr[:99], k=0.01)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
# ax.plot(xCopy[:, 1], yHat)

# ridgeWeight = ridgeTest(xArr, yArr)
# ridgeWeight = ridgeWeight[:, :]
# ax.plot(ridgeWeight)
# print(ridgeWeight)

# plt.show()

# yHat = xMat*ws
# yHat01 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], k=0.1)
# yHat1 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], k=1)
# yHat10 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], k=10)

# print(rssError(yArr[100:199], yHat01.T))
# print(rssError(yArr[100:199], yHat1.T))
# print(rssError(yArr[100:199], yHat10.T))

# ws = standRegres(xArr[0:99], yArr[0:99])
# yHat = np.mat(xArr[100:199])*ws
# print(rssError(yArr[100:199], yHat.T.A))

# corrcoef = np.corrcoef(yHat.T, yMat)
# print(corrcoef)

# print(np.eye((3)))

from time import sleep
import json
import urllib3

http = urllib3.PoolManager()


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    print(searchURL)
    pg = http.request('GET', searchURL)
    retDict = json.loads(pg.data)
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: 
            print('problem with item %d' % i)
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


lgX = []
lgY = []
setDataCollect(lgX, lgY)
