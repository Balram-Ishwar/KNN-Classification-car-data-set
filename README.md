# KNN-Classification-car-data-set
#Use Jupter-notebook to run this code



import numpy as np

import pandas as pd

from sklearn import neighbors, metrics

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

#loading of data

columns_name =['buying','maint','doors','persons','lug_boots','safety','class']
data = pd.read_csv('car.data',names=columns_name)
data.head()

X= data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]

# Converting the data from string to number of X

Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i]=Le.fit_transform(X[:, i])
   
# Converting the data from string to number of y using Mapping

label_mapping={
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class']=y['class'].map(label_mapping)
y =np.array(y)


#Creating model

knn = KNeighborsClassifier(n_neighbors=25, weights='uniform')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
knn.fit(X_train,y_train)
prediction =knn.predict(X_test)

accuracy=metrics.accuracy_score(y_test,prediction)

"prediction: ",prediction
"Accuracy: ",accuracy

#Check the Pediction with actual data
a=12

"Actual Value: ",y[a]
"Predicted Values: ",knn.predict(X)[a]
