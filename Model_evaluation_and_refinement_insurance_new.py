
#Model development, evaluation and refinement, insurance dataset

#1. import libraries
import pandas as pd
import numpy as np
import piplite
await piplite.install('seaborn')
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings('ignore')
# Example function that raises a warning
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
fxn() # No warning will be shown

#2. import the dataset and load it to pd df
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'

await download(filepath, "insurance.csv")
file_name="insurance.csv"
df = pd.read_csv(file_name,names=['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region', 'charges'])
df.head()




#3. data wrangling
df.shape #(2772, 7)
df=df.replace('?', np.nan) # '?' values will be replaced with null vals
df.isnull().sum()#age has 4 null values and smoker has 7
#replace missing data in age column with mean
avg_age=df['age'].astype('float').mean()
df['age']=df['age'].replace(np.nan,avg_age)
#replace missing data in smoker column with most frequent value
mf_smoker=df['smoker'].value_counts().idxmax()
df['smoker']=df['smoker'].replace(np.nan,mf_smoker)
df.dtypes #age should be float, smoker should be int
#cast age and smoker to numeric datatypes
df['age']=pd.to_numeric(df['age'].astype(float))
df['smoker']=pd.to_numeric(df['smoker'].astype(int))

'''
#4. EDA
#charges is target var
#Run exploratory data analysis (EDA) and identify the attributes that most affect the charges
df.corr()#smoker has the strongest correlation with charges (0.788783), then age (0.298622) and bmi(0.199846) 
#regplot for age
sns.regplot(x='age',y='charges', data=df) #weak positive linear relation
plt.show()
#regplot for bmi
sns.regplot(x='bmi',y='charges', data=df) #weak positive linear relation
plt.show()
# boxplot for smoker
sns.boxplot(x='smoker',y='charges', data=df)
plt.show() #no overlap,  higher charges are related to smokers=1
#as a part of EDA we could also check regplots for all  continuous vars, boxplots for all categorical vars
#we could conduct group by for all categorical vars and Pearson corr and P value for all continuous vars
#we could also check the quality of data for all vars, such as value_counts() and df.describe()
#CONCLUSION: smoker, age and bmi are the of the most interest at first sight

#5. model development
#Fit a linear regression model that may be used to predict the charges value,
#just by using the smoker attribute of the dataset. Print the  score of this model.

#5.1.a. SLR - with train-test split
x=df[['smoker']]
y=df[['charges']]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
slr=LinearRegression()
slr.fit(x_train,y_train)
yhat_slr=slr.predict(x_test)
#two scores are not the same (?!?)
print('Out-of-sample score for SLR is: ',slr.score(x_test,y_test)) #0.5546540022600863
print('Out-of-sample R2 score for SLR is: ',r2_score(yhat_slr,y_test)) #0.33711930321535144

#5.1.b. SLR - without train-test split
x=df[['smoker']]
y=df[['charges']]
slr=LinearRegression()
slr.fit(x,y)
yhat_slr=slr.predict(x)
#two scores are not the same (?!?)
print('In-sample  score for SLR is: ',slr.score(x,y)) #0.6221791718835359 
print('In-sample R2 score for SLR is: ',r2_score(yhat_slr,y)) #0.3927459400919767

#5.2.a. MLR - with train-test split
x=df.drop(columns=['charges'])
y=df[['charges']]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
mlr=LinearRegression()
mlr.fit(x_train,y_train)
yhat_mlr=mlr.predict(x_test)
#two scores are not the same (?!?)
print('Out-of-sample score for MLR is: ', mlr.score(x_test,y_test)) #0.6760582410387508
print('Out-of-sample R2 score for MLR is: ', r2_score(yhat_mlr,y_test)) #0.5880656133803033


#5.2.b. MLR - without train-test split
x=df.drop(columns=['charges'])
y=df[['charges']]
mlr=LinearRegression()
mlr.fit(x,y)
yhat_mlr=mlr.predict(x)
#two scores are not the same (?!?)
print('In-sample score for MLR is: ', mlr.score(x,y)) #0.7504063768213818
print('In-sample R2 score for MLR is: ', r2_score(yhat_mlr,y)) #0.667388723113119


#5.3.a. MLR with Polynomial Transofrmation using Pipeline with train-test split
#Create a training pipeline that uses StandardScaler(), PolynomialFeatures() and LinearRegression() to create the model
x=df.drop(columns=['charges'])
y=df[['charges']]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#create pipeline input
input=[('polynomial',PolynomialFeatures(degree=2)),('scale', StandardScaler()),('model', LinearRegression())]
#create Pipeline object
pipe=Pipeline(input)
#fit the Pipeline object
pipe.fit(x_train,y_train)
#make prediction
yhat_pipe=pipe.predict(x_test)
print('Out-of-sample R2 score for MLR with Polynomial Transofrmation using Pipeline is: ', r2_score(y_test,yhat_pipe)) #0.783264600936911

#5.3.b. MLR with Polynomial Transofrmation using Pipeline without train-test split
#Create a training pipeline that uses StandardScaler(), PolynomialFeatures() and LinearRegression() to create the model
x=df.drop(columns=['charges'])
y=df[['charges']]
#create pipeline input
input=[('polynomial',PolynomialFeatures(degree=2)),('scale', StandardScaler()),('model', LinearRegression())]
#create Pipeline object
pipe=Pipeline(input)
#fit the Pipeline object
pipe.fit(x,y)
#make prediction
yhat_pipe=pipe.predict(x)
print('In-sample R2 score for MLR with Polynomial Transofrmation using Pipeline is: ', r2_score(y,yhat_pipe)) #0.8451668430528109


#5.4.a. MLR model Ridge Regression (alpha = 0.1), no Polynomial Transformation of data - with train-test split
#define x and y
x=df.drop(columns=['charges'])
y=df[['charges']]
#train-test split the data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#create Ridge object with alpha = 0.1
RR=Ridge(alpha=0.1)
#fit the Ridge model with train data
RR.fit(x_train,y_train)
#predict on test data
yhat_rr=RR.predict(x_test)
print('Out-of-sample R2 score for MLR model Ridge Regression, no Polynomial Transformation of data is: ', r2_score(y_test,yhat_rr)) #0.6760802484653843


#5.4.b. MLR model Ridge Regression (alpha = 0.1), with Polynomial Transformation of data (2nd degree) - with train-test split
#Polynomially transform x data
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train) #BE CAREFUL, it is FIT TRANSFORM, not FIT
x_test_pr=pr.fit_transform(x_test) # BE CAREFUL, it is FIT TRANSFORM, not FIT
#create Ridge object with alpha = 0.1
RR_pr=Ridge(alpha=0.1)
#fit Ridge model with transformed data
RR_pr.fit(x_train_pr,y_train)
#predict on test data
yhat_rr_pr=RR_pr.predict(x_test_pr)
print('Out-of-sample R2 score for MLR model Ridge Regression, with Polynomial Transformation of data is: ', r2_score(y_test,yhat_rr_pr)) #0.7835630540960404

#5.5 We could further refine the last model by trying out different degrees of polynomials and by trying out different alphas
#so first we will search for a point of overfitting 
#create an empty list to store R2 scores
R2_list=[]
#create a list of degrees
degrees=[1,2,3,4,5]
for i in degrees:
    pr=PolynomialFeatures(degree=i)
    x_train_pr=pr.fit_transform(x_train) #BE CAREFUL, it is FIT TRANSFORM, not FIT
    x_test_pr=pr.fit_transform(x_test) # BE CAREFUL, it is FIT TRANSFORM, not FIT
    #we will not work with Ridge object now, but with MLR, to understand the relation between degree and R2
    #the other reason is that Ridge object displays error
    #results for both with alpha = 1, even 0.1 will be very similar to MLR results
    #RR_pr=Ridge(alpha=0.1) #or alpha=1
    MLR_pr=LinearRegression()
    #fit Ridge model with transformed data
    #RR_pr.fit(x_train_pr,y_train)
    MLR_pr.fit(x_train_pr,y_train)
    #predict on test data
    #yhat_rr_pr=RR_pr.predict(x_test_pr)
    yhat_mlr_pr=MLR_pr.predict(x_test_pr)
    R2_list.append(r2_score(y_test,yhat_mlr_pr)) 
R2_list

plt.plot(degrees,R2_list)
plt.title('Point of overfitting')
plt.xlabel('Degrees')
plt.ylabel('R2')
plt.show()#degree 4 of polynomial is the best according to the plot



#5.6 tunning the degree
#MLR model Ridge Regression (alpha = 0.1), with Polynomial Transformation degree 4 of data - with train-test split
#define x and y again so I can comment out all the lines above
x=df.drop(columns=['charges'])
y=df[['charges']]
#train-test split the data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#Transform the data
pr=PolynomialFeatures(degree=4)
x_train_pr=pr.fit_transform(x_train) #BE CAREFUL, it is FIT TRANSFORM, not FIT
x_test_pr=pr.fit_transform(x_test) # BE CAREFUL, it is FIT TRANSFORM, not FIT
#create Ridge object with alpha = 0.1
RR_pr=Ridge(alpha=0.1) #this returns an error when degree is 4
#fit Ridge model with transformed data
RR_pr.fit(x_train_pr,y_train)
#predict on test data
yhat_rr_pr=RR_pr.predict(x_test_pr)
print('Out-of-sample R2 score for MLR model Ridge Regression, with Polynomial Transformation of degree 4 is: ', r2_score(y_test,yhat_rr_pr)) #0.7911381226323921
'''
#5.7 tunning the alpha using GridSearchCV (I think the best alpha is between 5 and 15)
#MLR model with Polynomial Transformation degree 4 of data using GridSearchCV to find the best alpha - with train-test split
#define a list of alphas we want to test
parameters=[{'alpha':np.arange(5,15,2)}] #I choose step to be 1 because otherwise the execution takes too long
#define x and y again so I can comment out all the lines above
x=df.drop(columns=['charges'])
y=df[['charges']]
#train-test split the data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#Transform the data with polynomial of degree 4, which is the best one
pr=PolynomialFeatures(degree=4)
x_train_pr=pr.fit_transform(x_train) #BE CAREFUL, it is FIT TRANSFORM, not FIT
x_test_pr=pr.fit_transform(x_test) # BE CAREFUL, it is FIT TRANSFORM, not FIT
#create Ridge object 
RR_pr=Ridge() 
Grid1=GridSearchCV(RR_pr,parameters,cv=4)
Grid1.fit(x_train_pr,y_train)#returns an error if there is no filter warnings ignore specified above
best_alpha=Grid1.best_estimator_ #the estimator with the best alpha
print('The Ridge with the best alpha is: ', best_alpha)
all_scores=Grid1.cv_results_ # all R2 scores
best_score=best_alpha.score(x_test_pr,y_test)
print('Out-of-sample R2 score for MLR model Ridge Regression, with Polynomial Transformation of degree 4 is and best alpha is: ', best_score) #0.791509199713393
