#!/usr/bin/env python
# coding: utf-8

# Importing all the required libraries

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn import tree
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)
import os
from sklearn.pipeline import make_pipeline
os.getcwd()
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as s
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# Loading the datasets
train = pd.read_csv('aug_train.csv')
test = pd.read_csv('aug_test.csv')
sample_submission = pd.read_csv('sample_submission.csv')


#    
#    ### EDA -  Checking basic data charecteristics
# 
# 
# 

train.head()

train.shape

train.info()


train.nunique()

# City, Employee ID will NOT be used as Features since all these have high unique values


train.isnull().sum(axis = 0)

train.describe()


# ### Data Visualization


ax = sns.displot(train['gender']).tick_params(labelrotation=45);


sns.displot(train['relevent_experience']).tick_params(labelrotation=45);


sns.displot(train['enrolled_university']).tick_params(labelrotation=45);


sns.displot(train['education_level']).tick_params(labelrotation=45);


sns.displot(train['major_discipline']).tick_params(labelrotation=45);


sns.displot(train['company_size']).tick_params(labelrotation=45);


sns.displot(train['company_type']).tick_params(labelrotation=45);


# Checking frequency tables for other variables
train['experience'].value_counts()


train['last_new_job'].value_counts()


# Plotting histogram for numeric variable
plt.hist(train['training_hours']);


# Checking dependent variable split in data
train['target'].value_counts()


train['city'].value_counts() 

# City will be removed from this dataset as a feature as number of unique values are very high and 
# hence we will not consider it as a categorical variable


# Also removing 'enrollee_id' from data since it is not required in model
train.drop(['enrollee_id','city'],axis=1,inplace=True)


# ### Separating independent and dependent variables


X = train.drop(columns=['target'])

y = train['target']

X.columns

y.head()

# Encoding categorical variables
X_cats = (OneHotEncoder(sparse=False,handle_unknown='ignore')
                   .fit_transform(X[['gender','relevent_experience',
                           'enrolled_university','education_level',
                           'major_discipline','company_type',
                           'last_new_job','experience','company_size']]))
X_cats = pd.DataFrame(X_cats)


# Merging encoded categorical variables with numeric variables
X_numerical = X.drop(columns=['gender','relevent_experience',
                                  'enrolled_university','education_level',
                                 'major_discipline','company_type',
                                  'last_new_job','experience','company_size'])
col_names = X_numerical.columns
X_numerical = pd.DataFrame(X_numerical, columns=col_names)
X = X_numerical.join(X_cats)


# Since the Target has "0" far more than "1" we will just fill NA values with 0 for this problem
X.fillna(0, inplace=True)

y.value_counts().plot(kind='bar')


# Class imbalance is clearly visible
plt.figure(figsize=(6, 4))
y.value_counts().plot(kind='bar')
plt.ylabel('Number of rows', fontsize=12)
plt.xlabel('Target', fontsize=12)
plt.title('Before sampling')
plt.show()



# Using SMOTE for handling class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 0)
X.columns = X.columns.astype('str')

X_smote, y_smote = smote.fit_resample(X,y)


plt.figure(figsize=(6, 4))
y_smote.value_counts().plot(kind='bar')

plt.ylabel('Number of rows', fontsize=12)
plt.xlabel('Target', fontsize=12)
plt.title('After sampling')
plt.show()


# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_smote,
                                                    y_smote,
                                                    test_size=0.2,
                                                    random_state=42)


# Creating a function to test and compare various algorithms
def model_fit(x_train, y_train, test_data):
    
    #LogisticRegression
    alg = LogisticRegression(max_iter=1000)
    alg.fit(x_train, y_train)
    y_pred = alg.predict(test_data)  
    print('Logistic Regression Model 1')
    print('AUC On Test Set - {}'.format(roc_auc_score(y_pred, y_test)))
    conf_mat = confusion_matrix(y_pred, y_test)
    print('Confusion Matrix :\n',conf_mat)
    sensitivity1 = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
    print('Sensitivity : ', sensitivity1 )
    specificity1 = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
    print('Specificity : ', specificity1)
    f1_score2 = f1_score(y_test,y_pred)
    print('f1_score : ', f1_score2)
    
    #SVM
    alg = SVC()
    alg.fit(x_train, y_train)
    y_pred = alg.predict(test_data)  
    print('SVM Model 2')
    print('AUC On Test Set - {}'.format(roc_auc_score(y_pred, y_test)))
    conf_mat = confusion_matrix(y_pred, y_test)
    print('Confusion Matrix :\n',conf_mat)
    sensitivity1 = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
    print('Sensitivity : ', sensitivity1 )
    specificity1 = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
    print('Specificity : ', specificity1)
    f1_score2 = f1_score(y_test,y_pred)
    print('f1_score : ', f1_score2)
        
    #RandomForest
    alg = RandomForestClassifier()
    alg.fit(x_train, y_train)
    y_pred = alg.predict(test_data)  
    print('Random Forest Model 3')
    print('AUC On Test Set - {}'.format(roc_auc_score(y_pred, y_test)))
    conf_mat = confusion_matrix(y_pred, y_test)
    print('Confusion Matrix :\n',conf_mat)
    sensitivity1 = conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])
    print('Sensitivity : ', sensitivity1 )
    specificity1 = conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])
    print('Specificity : ', specificity1)
    f1_score2 = f1_score(y_test,y_pred)
    print('f1_score : ', f1_score2)
    

model_fit(X_train, y_train,X_test)


# #### From the above output, we can conclude that RandomForest is the best algorithm to go forward

# Tuning Hyperparameters
from sklearn.model_selection import RandomizedSearchCV

forest  = RandomForestClassifier(random_state = 42)

params = {
        'n_estimators' : [100, 300, 500, 800, 1200],
        'max_depth' : [5, 8, 15, 25, 30],
        'min_samples_split' : [2, 5, 10, 15, 100],
        'min_samples_leaf' : [1, 2, 5, 10] 
        }

gridF = RandomizedSearchCV(forest, params, cv = 5, verbose = 1)


clf_grid = gridF.fit(X_train, y_train)


clf_grid.best_params_

# Running model on entire data 
model = RandomForestClassifier(n_estimators=800,
                               min_samples_split=10,
                               min_samples_leaf=2, 
                               max_depth=30)
model.fit(X_smote, y_smote)



# Data managament on new test data same as training data
test.drop(['enrollee_id','city'],axis=1,inplace=True)

X_cats2 = (OneHotEncoder(sparse=False,handle_unknown='ignore')
                   .fit_transform(test[['gender','relevent_experience',
                           'enrolled_university','education_level',
                           'major_discipline','company_type',
                           'last_new_job','experience','company_size']]))
X_cats2 = pd.DataFrame(X_cats2)

X_numerical2 = test.drop(columns=['gender','relevent_experience',
                                  'enrolled_university','education_level',
                                 'major_discipline','company_type',
                                  'last_new_job','experience','company_size'])
col_names = X_numerical2.columns
X_numerical2 = pd.DataFrame(X_numerical2, columns=col_names)
X_test2 = X_numerical2.join(X_cats2)

X_test2.fillna(0, inplace=True)
X_test2.columns = X_test2.columns.astype('str')


# Prediction for new test data
sample_submission.target = model.predict(X_test2)
sample_submission.head()


sample_submission.to_csv('result.csv',index=False)

