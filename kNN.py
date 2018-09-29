#!/usr/bin/python

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def img2vector(filename,rownum,colnum):
	returnVect = zeros((1,rownum*colnum))
	fr = open(filename)
	for i in range(rownum):
		lineStr = fr.readline()
		for j in range(colnum):
			returnVect[0, colnum*i+j] = int(lineStr[j])
	return returnVect

def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		# classLabelVector.append(listFromLine[-1])
		index += 1
	return returnMat,classLabelVector

#D:\\MachineLearnWorkSpace\\machinelearninginaction\\Ch02\\datingTestSet2.txt
def showDemoImg(filename) :
	datingDataMat,datingLabels = file2matrix(filename)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
	plt.show()

# showDemoImg("D:\\MachineLearnWorkSpace\\machinelearninginaction\\Ch02\\datingTestSet2.txt")
# /home/lty/machine-learn/scripts/python-workspace/machinelearninginaction/Ch02/datingTestSet2.txt

def autoNorm(dataSet) :
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def datingClassTest(sourceFilePath,k,hoRatio):
	datingDataMat,datingLabels = file2matrix(sourceFilePath)
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],k)
		print "the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i])
		if(classifierResult != datingLabels[i]): errorCount+=1.0
	print "error count: %f, test count: %f, error rate: %f" % (errorCount, numTestVecs, (errorCount/float(numTestVecs)))

def classifyPerson(sourceFilePath):
	resultList = ['not at all','in small doses','in large doses']
	percentTats = float(raw_input("percentage of time spent playing video games?"))
	ffMiles = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters of ice cream consumed per year?"))
	datingDataMat,datingLabels = file2matrix(sourceFilePath)
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
	print "You will probably like this person:", resultList[classifierResult-1]

# classifyPerson("/home/lty/machine-learn/scripts/python-workspace/machinelearninginaction/Ch02/datingTestSet2.txt")

def handWritingClassTest(trainSourceDir, testSourceDir, rownum, colnum, k):
	hwLabels = []
	trainingFileList = listdir(trainSourceDir)
	m = len(trainingFileList)
	trainingMat = zeros((m, rownum*colnum))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('%s/%s'%(trainSourceDir,fileNameStr),rownum,colnum)
	testFileList = listdir(testSourceDir)
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('%s/%s'%(testSourceDir,fileNameStr), rownum, colnum)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
		print "the classifier came back with: %d, the real answer is: %d"%(int(classifierResult), classNumStr)
		if (classifierResult != classNumStr): errorCount += 1.0
	print "\nerror count: %d total test count:%d" % (errorCount,mTest)
	print "\nthe total error rate is: %f"%(errorCount/float(mTest))

# handWritingClassTest("/home/lty/machine-learn/scripts/python-workspace/machinelearninginaction/Ch02/tmp/trainingDigits"
# 	,"/home/lty/machine-learn/scripts/python-workspace/machinelearninginaction/Ch02/tmp/testDigits"
# 	,32,32,3)
