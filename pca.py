import numpy as np




file_path_pattern = "/home/lty/machine-learn-space/scripts/python-workspace/machinelearninginaction/Ch13/{}"
test_set_path = file_path_pattern.format('testSet.txt')
secom_set_path = file_path_pattern.format('secom.data')

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    print("mean Val:{}".format(meanVals))
    meanRemoved = dataMat - meanVals
    print("mean removed:{}".format(meanRemoved))
    covMat = np.cov(meanRemoved, rowvar=False)
    print("cov matrix:{}".format(covMat))
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print("eig vals:{}".format(eigVals))
    print("eig vectors:{}".format(eigVects))
    # print("shape of eig vals:{}".format(np.shape(eigVals)))
    # print("shape of eig vectors:{}".format(np.shape(eigVects)))
    eigValInd = np.argsort(eigVals)
    print("eigValInd:{}".format(eigValInd))
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    print("eigValInd:{}".format(eigValInd))
    redEigVects = eigVects[:, eigValInd]
    print("redEigVects:{}".format(redEigVects))
    lowDDataMat = meanRemoved * redEigVects
    print('lowDDataMat:{}'.format(lowDDataMat))
    reconMat = (lowDDataMat*redEigVects.T) + meanVals
    print('reconMat:{}'.format(reconMat))
    return lowDDataMat, reconMat


def replaceNanWithMean():
    dataMat = loadDataSet(secom_set_path, ' ')
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i].A))[0], i])
        dataMat[np.nonzero(np.isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat


dataMat = replaceNanWithMean()
meanVals = np.mean(dataMat)
meanRemoved = dataMat - meanVals

covMat = np.cov(meanRemoved, rowvar=False)
eigVals, eigVects = np.linalg.eig(np.mat(covMat))

max_vectors = 30
eigValInd = np.argsort(eigVals)[:-(max_vectors+1):-1]
topEigVals = eigVals[eigValInd]

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(range(max_vectors), topEigVals, color='blue', linewidth=2.0, linestyle='--', marker='^')

plt.show()

# # x = np.array([[4, 2, -5], [6, 4, -9], [5, 3, -7]])
# # x = np.array([[-1, -1, 0, 2, 0], [-2, 0, 0, 1, 1]])
# # x = np.array([[1, -2], [2, -3]])
# # dataMat = np.mat(x)

# dataMat = loadDataSet(test_set_path)
# print("data Mat:{}".format(dataMat))
# lowDMat, reconMat = pca(dataMat, 1)
# print(np.shape(lowDMat))

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
# ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')

# plt.show()

# 
# x = np.array([[0, 2], [1, 1]])
# x = np.array([[0, 2], [1, 1], [2, 0]])
# print(x)
# print(np.cov(x, rowvar=True))

# 0 1
# 2 1
# 求协方差
# mean-x1=(0+1)/2=0.5   mean-x2=(2+1)/2=1.5
# cov(x1, x1) = [(0-0.5, 1-0.5)(0-0.5, 1-0.5)]/(2-1) = 0.25 + 0.25 = 0.5
# cov(x1, x2) = [(0-0.5, 1-0.5)(2-1.5, 1-1.5)]/(2-1) = -0.25 - 0.25 = -0.5
# cov(x2, x2) = [(2-1.5, 1-1.5)(2-1.5, 1-1.5)]/(2-1) = 0.25 + 0.25 = 0.5
# 最终得
# 0.5  -0.5
# -0.5  0.5

# 0 2
# 1 1
# 求协方差
# mean-x1=(0+2)/2=1 mean-x2=(1+1)/2 = 1
# cov(x1, X1) = [(0-1, 2-1)(0-1, 2-1)]/(2-1) = 1+1 = 2
# cov(x1, x2) = [(0-1, 2-1)(1-1, 1-1)]/(2-1) = 0
# cov(x2, x2) = [(1-1, 1-1)(1-1, 1-1)]/(2-1) = 0
# 最终得
# 2  0
# 0  0

# 0 1 2
# 2 1 0
# 求协方差
# 按列求协方差
# mean-x1 = (0+1+2)/3 = 1    mean-x2 = (2+1+0)/3 = 1
# cov(x1, x1) = [(0-1, 1-1, 2-1)(0-1, 1-1, 2-1)]/(3-1) = (1 + 0 + 1)/2 = 1
# cov(x1, x2) = [(0-1, 1-1, 2-1)(2-1, 1-1, 0-1)]/(3-1) = (-1 + 0 + -1)/2 = -1
# cov(x2, x2) = [(2-1, 1-1, 0-1)(2-1, 1-1, 0-1)]/(3-1) = (1 + 0 + 1)/2 = 1
# 最终得
#  1  -1
# -1   1




