import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


DataSet=pd.read_csv("iris.csv")
x=DataSet.iloc[:,:-1].values
y=DataSet.iloc[:,-1].values

from sklearn.tree import DecisionTreeClassifier as DTC
dtree=DTC(criterion="entropy")

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


model=dtree.fit(x_train,y_train)

y_pred=dtree.predict(x_test)
print(y_pred)

print("Confusion matrix= ",confusion_matrix(y_test,y_pred))
print("Accuracy score= ",accuracy_score(y_test,y_pred)*100)

import graphviz as gv
import sklearn.tree as tree

GV_compute=tree.export_graphviz(model,feature_names=["sepal.length","sepal.width","petal.length","petal.width"],class_names=["Setosa","Versicolor","Virginica"])

X1=gv.Source(GV_compute)
X1.render("Dtree_iris2")
