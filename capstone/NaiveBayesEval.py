import pandas as pd
print("Reading X")
dataX=pd.read_csv('/Users/rparundekar/dataspace/dbpedia2016/datasetX.csv')
dataX.head()
print("Reading Y")
dataY=pd.read_csv('/Users/rparundekar/dataspace/dbpedia2016/datasetY.csv')
dataY.head()
print("Done Reading")

del dataX['id']
del dataY['id']

import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.20, random_state=42)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

conf = [[0,0],[0,0]];

for classY in y_train:
    print("Learning & Testing for " + classY )
    y_train_this=y_train[classY]
    y_test_this=y_test[classY]
    clf = MultinomialNB()
    clf.fit(X_train, y_train_this)
    y_pred=clf.predict(X_test)
    conf = conf + confusion_matrix(y_test_this, y_pred)
    print (conf)
    
tp = (float)(conf[1,1])
fp = (float)(conf[0,1])
fn = (float)(conf[1,0])
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1score = 2.0 * precision * recall / (precision+recall)

print ('Precision  : %.2f',precision*100)
print ('Recall     : %.2f',recall*100)
print ('F1 Score   : %.2f',f1score*100)



