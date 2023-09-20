import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# this script include "linear regression" and "Logistic regression"

#######################
## Linear regression ##
#######################
folder = 'Refactored_Py_DS_ML_Bootcamp-master\\11-Linear-Regression\\'
USAhousing = pd.read_csv(folder+'USA_Housing.csv')
USAhousing.head()
USAhousing.info()
USAhousing.describe()
USAhousing.columns

## EDA
sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr())

## Train Test split
from sklearn.model_selection import train_test_split
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
X_train.shape; X_test.shape

## Train LM model
from sklearn.linear_model import LinearRegression
lm = LinearRegression() # try fit_intercept=False
lm.fit(X_train,y_train)

## model evaluation
lm.intercept_, lm.coef_
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient']); coeff_df
# X.columns == lm.feature_names_in_

## predictions
predictions = lm.predict(X_test)
1-sum((predictions - y_test)**2) / sum((np.mean(y_test) - y_test)**2) # R^2
plt.scatter(y_test,predictions); plt.show()
sns.distplot((y_test-predictions),bins=50);

# evaluation metrics
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



#########################
## Logistic regression ##
#########################
folder = 'Refactored_Py_DS_ML_Bootcamp-master\\13-Logistic-Regression\\'
train = pd.read_csv(folder+'titanic_train.csv')
train.head()
train.info() # NaN in the data
np.sum(train.isna()) # NaN in the data
train.Cabin

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
plt.show()

## filling mussing
def impute_age(cols):
    Age, Pclass = cols
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
any(train['Age'].isna())

train.drop('Cabin',axis=1,inplace=True) # remove column due to too many missings
train.head(); np.sum(train.isna()) # Embarked still has 2 na
train.dropna(inplace=True) # remove two rows

train.info()
train.Sex.value_counts(); train.Embarked.value_counts()
sex = pd.get_dummies(train['Sex'],drop_first=True) # drop_first=True -> remove the first level
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True) # remove 4 columns
train = pd.concat([train,sex,embark],axis=1)

## Buliding logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

## Evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
                                                    
sum((y_test==predictions) & (y_test==0))/sum((y_test==predictions)) # sensitivity
sum((y_test==predictions)[y_test==1])/sum((y_test==1))
sum((y_test==predictions)[y_test==0])
