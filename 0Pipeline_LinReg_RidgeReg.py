# Polynomial and ridge regression pipelines and grid search  - insurance dataset


import piplite
await piplite.install('seaborn')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
await download(filepath, "insurance.csv")
file_name="insurance.csv"
df = pd.read_csv(file_name, header=None)
df.columns = ['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region', 'charges']
df.head()
df.replace('?',np.nan, inplace=True)
avg_age=df['age'].astype('float').mean()
df['age']=df['age'].replace(np.nan,avg_age)
mode_smoker=df['smoker'].value_counts().idxmax()
df['smoker']=df['smoker'].replace(np.nan, mode_smoker)
df.isnull().sum()
df['age'] = df['age'].astype('float')
df['charges']=df['charges'].round(2)

df.corr() #smoker is the most important feature, but we will consider them all

#1.develop polynomial regression model using pipeline 
y=df['charges']
x=df.drop('charges',axis=1)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)
#when polynomial orders is not specified, the default is 2
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
lr_pipe=Pipeline(Input)
lr_pipe.fit(x_train, y_train)
#prediction on test set
y_pred_pipe=lr_pipe.predict(x_test)
print('R2 score of Polynomial Regression Pipeline:',r2_score(y_test,y_pred_pipe)) #R2 is 0.8339352280040062


#2.develop ridge regression model using pipeline 
#when alpha is not specified, it will be assigned 1 by default
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', Ridge())]
ridge_pipe=Pipeline(Input)
ridge_pipe.fit(x_train, y_train) 
#prediction on test set
y_pred_ridge = ridge_pipe.predict(x_test)
print('R2 score of Ridge Pipeline:',r2_score(y_test,y_pred_ridge)) #R2 is 0.8339413909195286  
#refine the model using grid search to find the best degree (2 or 3) and the best alpha between 0 and 0.2, increasing alpha for 0.01
parameters = [ {'polynomial__degree': [2, 3], 
                'model__alpha': np.arange(0, 0.2, 0.01) } ]
#we pass the esitmator ridge_pipe and, parameters and the number of folds
grid_search = GridSearchCV(estimator = ridge_pipe, 
                           param_grid = parameters,
                           cv = 4)
grid_search = grid_search.fit(x_train, y_train) 
print('Grid Search using Ridge Regression as Estimator')
print("Best alpha:", grid_search.best_params_['model__alpha'])#best alpha is 0.19
print("Best polynomial degree:", grid_search.best_params_['polynomial__degree'])#best degree is 3
print("Best R2 score with the above best parameters:", grid_search.best_score_) #R2 is 0.8468684169243493
