#!usr/bin/env python
# -*- coding:utf-8 -*-
from numpy import *
import operator
import csv
def nomalizing(array):
	m,n=shape(array)
	for i in xrange(m):
		for j in xrange(n):
			if array[i,j]!=0:
				array[i,j]=1
	return array

def toInt(array):
	array=mat(array)
	m,n=shape(array)
	newArray=zeros((m,n))
	for i in xrange(m):
		for j in xrange(n):
				newArray[i,j]=int(array[i,j])
	return newArray

def loadTrainData():
	l=[]
	with open('train.csv') as file:
		lines= csv.reader(file)
		for line in lines:
			l.append(line)
	l.remove(l[0])
	l=array(l)
	label=l[:,0]
	data=l[:,1:]
	return nomalizing(toInt(data)),toInt(label)

def loadTestData():
	l=[]
	with open('test.csv') as file:
		lines = csv.reader(file)
		for line in lines:
			l.append(line)
	l.remove(l[0])
	data=array(l)
	return nomalizing(toInt(data))

def classify(inX, dataSet, labels, k):
    inX=mat(inX)
    dataSet=mat(dataSet)
    labels=mat(labels)
    dataSetSize = dataSet.shape[0]                  
    diffMat = tile(inX, (dataSetSize,1)) - dataSet   
    sqDiffMat = array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)                  
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()            
    classCount={}                                      
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i],0]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def saveResult(result,fileName):
	with open(fileName,'wb') as myFile:
		myWriter=csv.writer(myFile)
		tmp=['ImageId','Label']
		myWriter.writerow(tmp)
		j=0
		for i in result:
			j+=1
			tmp=[]
			tmp.append(j)
			tmp.append(int(i))
			myWriter.writerow(tmp)

#调用scikit的knn算法包
from sklearn.neighbors import KNeighborsClassifier  
def knnClassify(trainData,trainLabel,testData): 
    knnClf=KNeighborsClassifier()#default:k = 5,defined by yourself:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData,ravel(trainLabel))
    testLabel=knnClf.predict(testData)
    saveResult(testLabel,'sklearn_knn_Result.csv')
    return testLabel

#调用scikit的SVM算法包
from sklearn import svm   
def svcClassify(trainData,trainLabel,testData): 
    svcClf=svm.SVC(C=5.0) #default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’  
    svcClf.fit(trainData,ravel(trainLabel))
    testLabel=svcClf.predict(testData)
    print testLabel
    saveResult(testLabel,'sklearn_SVC_C=5.0_Result.csv')
    return testLabel

def handwritingClass():
	print 'go'
	trainData,tarinLabel=loadTrainData()
	testData=loadTestData()
	knnClassify(trainData,tarinLabel,testData)
	# m,n=shape(testData)
	# resultList=[]
	# for i in range(m):
	# 	classifierResult=classify(testData[i],trainData,tarinLabel.transpose(),5)
	# 	print classifierResult
	# 	resultList.append(classifierResult)
	# saveResult(resultList,'result.csv')
	print 'end'

handwritingClass()