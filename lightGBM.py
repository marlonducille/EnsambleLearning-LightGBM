# C:\Users\Karen\AppData\Local\Programs\Python\Python38\Scripts
# C:\Users\Karen\anaconda3\Scripts

import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame 
import lightgbm as lgb 
import datetime

#loading our training dataset 'adult.csv' 
data=pd.read_csv('adult.csv',header=None) 

data.columns = ['age','workclass','fnlwgt','education','education-num','marital_Status','occupation','relationship',
                'race','sex','capital_gain','capital_loss','hours_per_week','native_country','Income']

# Label Encoding our target variable 
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
l.fit(data.Income)
data.Income = Series(l.transform(data.Income))

#label encoding our target variable 
data.Income.value_counts()

#One Hot Encoding of the Categorical features 
one_hot_workclass=pd.get_dummies(data.workclass) 
one_hot_education=pd.get_dummies(data.education) 
one_hot_marital_Status=pd.get_dummies(data.marital_Status) 
one_hot_occupation=pd.get_dummies(data.occupation) 
one_hot_relationship=pd.get_dummies(data.relationship) 
one_hot_race=pd.get_dummies(data.race) 
one_hot_sex=pd.get_dummies(data.sex) 
one_hot_native_country=pd.get_dummies(data.native_country) 

#removing categorical features 
data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True)

#Merging one hot encoded features with our dataset 'data' 
data=pd.concat([data,one_hot_workclass,one_hot_education,one_hot_marital_Status,one_hot_occupation,
                one_hot_relationship,one_hot_race,one_hot_sex,one_hot_native_country],axis=1) 

#removing dulpicate columns  
_, i = np.unique(data.columns, return_index=True) 
data=data.iloc[:, i] 

#Here our target variable is 'Income' with values as 1 or 0.  
#Separating our data into features dataset x and our target dataset y 
x=data.drop('Income',axis=1) 
y=data.Income 

#Imputing missing values in our target variable 
y.fillna(y.mode()[0],inplace=True)

#Now splitting our dataset into test and train 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

train_data=lgb.Dataset(x_train,label=y_train)

#setting parameters for lightgbm
param = {'num_leaves':150, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':200} 
param['metric'] = ['auc', 'binary_logloss']

#Here we have set max_depth in xgb and LightGBM to 7 to have a fair comparison between the two.
#training our model using light gbm
num_round=50 
start=datetime.datetime.now() 
lgbm=lgb.train(param,train_data,num_round) 
stop=datetime.datetime.now()

#predicting on test set
ypred=lgbm.predict(x_test)

#converting probabilities into 0 or 1
for i in range(0,9769):    
  if ypred[i]>=.5:       # setting threshold to .5       
     ypred[i]=1    
  else:         
    ypred[i]=0

#calculating accuracy
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy_lgbm = accuracy_score(ypred, y_test) 

#calculating roc_auc_score for light gbm
auc_lgbm  = roc_auc_score(y_test, ypred)






