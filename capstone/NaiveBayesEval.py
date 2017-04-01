import pandas as pd

print("Reading Header Y")
headerY=pd.read_csv('/Users/rparundekar/dataspace/dbpedia2016/headerY.csv')

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

#Folder for the dataset
datasetFolder = '/Users/rparundekar/dataspace/dbpedia2016/dataset/'
#Number of files
numberOfFiles = 52
#Test split
testSplit=0.25

#Randomize the list of numbers so we can split train and test dataset
listOfFiles=list(range(1,numberOfFiles+1))
import random
random.shuffle(listOfFiles)
splitIndex=int((1-testSplit)*numberOfFiles)

confusionMatrix = [[0,0], [0,0]]

#Iterate through each class for creating the NB for that class
for index, row in headerY.iterrows():
    classId=row[0]
    classY=row[1]
    if(classId=='id'):
        continue
    print("Learning for " + classId )
    #Initialize the classifier
    classifier = linear_model.SGDClassifier()#GaussianNB()
    for trainIndex in range(0,splitIndex):
        print('Reading X for file datasetX_{}.csv'.format(listOfFiles[trainIndex]))
        dataX=pd.read_csv(datasetFolder + 'datasetX_{}'.format(listOfFiles[trainIndex]) + '.csv')
        
        print('Reading Y for file datasetY_{}.csv'.format(listOfFiles[trainIndex]))
        dataY=pd.read_csv(datasetFolder + 'datasetY_{}'.format(listOfFiles[trainIndex])  + '.csv')
        
        del dataX['id']
        del dataY['id']

        y_train_this=dataY[classY]
        print('Updating model')
        classifier.partial_fit(dataX, y_train_this, classes=np.array([0,1]))
    
    currentConfusionMatrix = [[0,0], [0,0]]
    print("Testing for " + classId)
    for testIndex in range(splitIndex,numberOfFiles):
        print('Reading X for file datasetX_{}.csv'.format(listOfFiles[testIndex]) )
        dataX=pd.read_csv(datasetFolder + 'datasetX_{}'.format(listOfFiles[testIndex]) + '.csv')
        
        print('Reading Y for file datasetY_{}.csv'.format(listOfFiles[testIndex]))
        dataY=pd.read_csv(datasetFolder + 'datasetY_{}'.format(listOfFiles[testIndex])  + '.csv')
        
        del dataX['id']
        del dataY['id']

        y_test_this=dataY[classY]
        print('Predicting Y' )
        y_pred = classifier.predict(dataX)
        currentConfusionMatrix = currentConfusionMatrix + confusion_matrix(y_test_this, y_pred)
    
    print('Class ' + classId + ' confusion matrix: ')
    print(currentConfusionMatrix)   
    confusionMatrix=confusionMatrix+currentConfusionMatrix
    
print('Complete confusion matrix: ')
print(confusionMatrix)   



tp = (float)(confusionMatrix[1,1])
fp = (float)(confusionMatrix[0,1])
fn = (float)(confusionMatrix[1,0])
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2.0 * precision * recall / (precision+recall)

print ('Precision  : %.2f',precision*100)
print ('Recall     : %.2f',recall*100)
print ('F1 Score   : %.2f',f1score*100)



