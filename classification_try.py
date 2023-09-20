import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# this script include K-nearest, decision tree & random forest, SVM
folder = 'Refactored_Py_DS_ML_Bootcamp-master\\14-K-Nearest-Neighbors\\'
df = pd.read_csv(folder+"Classified Data",index_col=0)
df.head


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

#########
## KNN ##
#########
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
## use for loop to choose K by seeing its prediction accuracy

###################
## Decision tree ##
###################
from sklearn.tree import DecisionTreeClassifier
folder = 'Refactored_Py_DS_ML_Bootcamp-master\\15-Decision-Trees-and-Random-Forests\\'
df = pd.read_csv(folder+'kyphosis.csv')
df.head()
X = df.drop('Kyphosis',axis=1); y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred = dtree.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

# Tree visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[1:])
features
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  


###################
## Random Forest ##
###################
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

pred = rfc.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))


############################
## support vector machine ##
############################
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
cancer = load_breast_cancer()
cancer.keys()
cancer.data.shape
print(cancer.DESCR)
cancer['feature_names']

## set dataframe
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
df_feat.info()

# np.ravel(df_target) return 1d array
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)
model = SVC()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

## Gridsearch: Finding the right parameters (like what C or gamma values to use)
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
pred = grid.best_estimator_.predict(X_test) # to grid.predict
pred2 = grid.predict(X_test)

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred2))
print(confusion_matrix(y_test,pred2))
