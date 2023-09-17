#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Classification model demonstration: Logistic Regression Model & Random Forest Model  
# Project background: use the IBM.csv dataset, which contains information about employees, to indicate whether they left the company (Attrition).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import statsmodels.api as sm
import numpy as np 


# # Logistic Regression Model
# Import the dataset 
IBM =pd.read_csv('IBM.csv')
IBM.head()
IBM.describe()
IBM.info()

# Numeric variables: 'Age', 'DistanceFromHome',  'MonthlyIncome', 'NumCompaniesWorked', 'YearsAtCompany'
# Categorical variables: 'Attrition', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'JobSatisfaction', 'MaritalStatus', 'WorkLifeBalance'

#Distribuution of outcome variable "Attrition"
IBM.Attrition.value_counts()

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(IBM['Attrition'])
IBM['Attrition'] = label_encoder.transform(IBM['Attrition'])

# Check missing values
missing_values = IBM.isnull().sum()
missing_values

# Build a correlation matrix
selected_vars = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'YearsAtCompany']
subset_df = IBM[selected_vars]
correlation_matrix = subset_df.corr().round(2)

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Dummify selected variables
IBM_dummies = pd.get_dummies(IBM, drop_first=True, columns=['Department', 'Education','EducationField','EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'MaritalStatus']) 

# Create a data partition, make 40%/60% split for test and training sets
X = IBM_dummies[['Age', 'DistanceFromHome', 'EnvironmentSatisfaction_2',
       'EnvironmentSatisfaction_3', 'EnvironmentSatisfaction_4',
       'JobSatisfaction_2', 'JobSatisfaction_3', 'JobSatisfaction_4', 'MonthlyIncome', 'NumCompaniesWorked',
       'YearsAtCompany', 'Department_Research & Development',
       'Department_Sales', 'Education_2', 'Education_3', 'Education_4',
       'Education_5', 'EducationField_Life Sciences',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Other', 'EducationField_Technical Degree',
       'WorkLifeBalance_2', 'WorkLifeBalance_3', 'WorkLifeBalance_4',
       'MaritalStatus_Married', 'MaritalStatus_Single']]
y=IBM_dummies[['Attrition']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=829)

print(X_train.shape, X_test.shape, type(X_train), type(X_test))
print(y_train.shape, y_test.shape, type(y_train), type(y_test))

# Compare the mean values of the variables in the dataset after grouping by 'Attrition'
IBM_dummies.groupby('Attrition').mean()

# MonthlyIncome: This variable appears to have a strong impact on the outcome. The mean values of this variable are significantly higher under class "0" compared to class "1" (6832.74 vs 4787.09). It is likely that employees with higher salaries are more inclined to stay at IBM as they feel satisfied about the treatment, and vice versa.
# YearsAtCompany: This variable could also have an impact on the outcome. The mean values of this variable are higher under class "0" compared to class "1" (7 vs 5). It is likely that employees who have spent more years with IBM are less likely to leave due to their familiarity with the company culture and loyalty, and vice versa.
# NumCompaniesWorked: It seems that the number of companies employees have previously worked for does not have a significant impact, as the mean values for both classes are very close (with only a 0.3 difference). 

# Run logistic regression
logit_model=sm.Logit(y_train, sm.add_constant(X_train))
result=logit_model.fit()
print(result.summary())

# Numeric variables showing high p-values: YearsAtCompany
# Categorical variables showing high p-values for ALL of the levels: Department, Education, EducationField

# Drop high pvalue variables and run logistic regression again 
X_train2 = X_train.drop(['YearsAtCompany', 'Department_Research & Development', 'Department_Sales', 'Education_2','Education_3','Education_4','Education_5','EducationField_Life Sciences','EducationField_Marketing','EducationField_Medical','EducationField_Other','EducationField_Technical Degree'], axis=1)
logit_model = sm.Logit(y_train, sm.add_constant(X_train2))
result2 = logit_model.fit()
print(result2.summary())

# LLR p-value is smaller for the second model comparing to the first model (8.794e-19 < 3.164e-18)

# Build model
logmodel = LogisticRegression()
logmodel.fit(X_train2, y_train)
LogisticRegression()
# Make predictions and assess the performance against the train set
from sklearn.metrics import accuracy_score
predictions1 = logmodel.predict(X_train2)
accuracy = accuracy_score(y_train, predictions1)
print("Accuracy against the train set:", accuracy)

# Make predictions and assess the performance against the test set
X_test2 = X_test.drop(['YearsAtCompany', 'Department_Research & Development', 'Department_Sales', 'Education_2','Education_3','Education_4','Education_5','EducationField_Life Sciences','EducationField_Marketing','EducationField_Medical','EducationField_Other','EducationField_Technical Degree'], axis=1)
predictions2 = logmodel.predict(X_test2)
accuracy = accuracy_score(y_test, predictions2)
print("Accuracy against the test set:", accuracy)

# Build confusion matrix for test dataset
mat2=confusion_matrix(predictions2, y_test)
sns.heatmap(mat2, square=True, fmt = 'g', annot=True, cbar=False)
plt.xlabel('actual result')
plt.ylabel('predicted result')
a,b = plt.ylim()
a+=.5
b-=.5
plt.ylim(a,b)
plt.show()

from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score
# Assess the performance of the model against the test set with more metrics
accuracy = round(accuracy_score(y_test, predictions2), 3)

# Calculate sensitivity (recall)
sensitivity = round(recall_score(y_test, predictions2), 3)

# Calculate specificity
specificity = round(recall_score(y_test, predictions2, pos_label=0), 3)

# Calculate precision
precision = round(precision_score(y_test, predictions2), 3)

# Calculate balanced accuracy
balanced_accuracy = round(balanced_accuracy_score(y_test, predictions2), 3)

print('Answering N-a,b,c,d,e:')
print("Accuracy rate is", accuracy)
print("Sensitivity rate is", sensitivity)
print("Specificity rate is", specificity)
print("Precision rate is", precision)
print("Balanced Accuracy rate is", balanced_accuracy)

# Compare model’s accuracy against the training set vs. accuracy against the
accuracy = accuracy_score(y_train, predictions1)
print("Accuracy against the train set:", accuracy)
accuracy = accuracy_score(y_test, predictions2)
print("Accuracy against the test set:", accuracy)


#Comparing these two values is to determine if the model is overfitting or underfitting. 
#If the model's accuracy on the training set is significantly higher than the accuracy on the test set, it suggests that the model is overfitting. 

# Make a prediction for John
John = pd.DataFrame({
    'Age': [33],
    'DistanceFromHome': [28],
    'EnvironmentSatisfaction_2': [0],
    'EnvironmentSatisfaction_3': [1],
    'EnvironmentSatisfaction_4': [0],
    'JobSatisfaction_2': [1],
    'JobSatisfaction_3': [0],
    'JobSatisfaction_4': [0],
    'MonthlyIncome': [3000],
    'NumCompaniesWorked': [2],
    'WorkLifeBalance_2': [0],
    'WorkLifeBalance_3': [0],
    'WorkLifeBalance_4': [1],
    'MaritalStatus_Married': [1],
    'MaritalStatus_Single': [0]
})
prediction_John = logmodel.predict(John)
print("Prediction:", prediction_John)
# Based on the model, John will NOT leave.

# Get the predicted probabilities for John leaving
proba_John = logmodel.predict_proba(John)
probability_leave = proba_John[0][1]
print("Probability of leaving:", probability_leave)
# According to this model, the probability of John leaving the company is 29%. 


# # Random Forest Model

# Read the dataframe 
IBM2 =pd.read_csv('IBM.csv')
# convert ‘Attrition’ into a binary format
label_encoder = LabelEncoder()
label_encoder.fit(IBM2['Attrition'])
IBM2['Attrition'] = label_encoder.transform(IBM2['Attrition'])

# Dummify the categorical inputs
IBM2_dummies = pd.get_dummies(IBM2, columns=['Department', 'Education','EducationField','EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'MaritalStatus']) 
#Re-partition the data
X = IBM2_dummies[['Age', 'DistanceFromHome', 'MonthlyIncome',
       'NumCompaniesWorked', 'YearsAtCompany', 'Department_Human Resources',
       'Department_Research & Development', 'Department_Sales', 'Education_1',
       'Education_2', 'Education_3', 'Education_4', 'Education_5',
       'EducationField_Human Resources', 'EducationField_Life Sciences',
       'EducationField_Marketing', 'EducationField_Medical',
       'EducationField_Other', 'EducationField_Technical Degree',
       'EnvironmentSatisfaction_1', 'EnvironmentSatisfaction_2',
       'EnvironmentSatisfaction_3', 'EnvironmentSatisfaction_4',
       'JobSatisfaction_1', 'JobSatisfaction_2', 'JobSatisfaction_3',
       'JobSatisfaction_4', 'WorkLifeBalance_1', 'WorkLifeBalance_2',
       'WorkLifeBalance_3', 'WorkLifeBalance_4', 'MaritalStatus_Divorced',
       'MaritalStatus_Married', 'MaritalStatus_Single']]
y = IBM2_dummies['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=829)

import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'max_features': [8, 12, 16],
    'min_samples_leaf': [4, 6, 10]
}

# Create the random forest classifier
rf_model = RandomForestClassifier()

# Create GridSearchCV with the random forest model and parameter grid
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(grid_search.best_params_)

# Use best hyperparameters to fit the model
rf_model=RandomForestClassifier(n_estimators=100, max_depth=10, max_features=16, min_samples_leaf=4, random_state=829)
rf_model.fit(X_train,y_train)

# Feature importance
feature_imp_df = pd.DataFrame(list(zip(rf_model.feature_importances_, X_train)))
feature_imp_df.columns = ['feature importance', 'feature']
feature_imp_df = feature_imp_df.sort_values(by='feature importance', ascending=False)
feature_imp_df

# “MonthlyIncome” had the highest importance score (0.176464), indicating that it is the most important variable in predicting attrition in this dataset. "Age" was ranked as the second higest importance score (0.148593). "EducationField_Human Resources" and "EducationField_Other" were the lowest scored (0.000133) and the second lowest scored (0.000292). 

# Building a classification report
pred = rf_model.predict(X_test)
print(classification_report(y_test, pred))
# Accuracy rate: 0.86       
# Sensitivity rate: 0.12
# Specificity rate: 0.98      
# Precision: 0.58      
# Balanced accuracy: (0.12+0.98)/2 = 0.55

#Compare the random forest model's accuracy against training set vs. test set
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)
print("Accuracy on training set:", round(train_accuracy, 2))
print("Accuracy on test set:", round(test_accuracy, 2))

# X. Accuracy on training set is slightly higher than Accuracy on test set (0.03). 

# By using these models, HR can proactively identify employees who are at a higher risk of leaving the company and take targeted actions to retain them. By having insights into the factors that contribute to employee attrition, HR can take preventive measures to create a more fulfilling work environment and foster employee loyalty. For example, the logistic regression model indicates that higher levels of job satisfaction, environment satisfaction, and work-life balance, as well as having a family life, have a positive influence on employee engagement and decrease the possibility of attrition. This information can support HR's decision-making process, enhance retention efforts, and make recruitment more efficient by targeting candidates who exhibit such desirable characteristics and potentially can hire more stable candidates. Also, HR could be focused on selling elements the company is good at and what employees care during recruitment to attract talents. 
# One interesting finding is that the coefficient relationship between work-life balance level and attrition can be interesting: level 4 (best work-life-balance) contribute less to employee retention than level 2 (good) and level 3 (better). This might indicate that employees also don't feel engaged when the work and life is too balanced to the point that they feel don't have enough thing to do/contribute to the success of the company. HR could share such business findings with senior management team for better engagement and team management. 
# Fing based on the coefficient value from logistic model: 
# WorkLifeBalance_2            -0.7865    
# WorkLifeBalance_3            -0.9355      
# WorkLifeBalance_4            -0.5740  
