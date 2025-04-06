#Model Development

import pandas as pd
import piplite
await piplite.install('seaborn')
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from pyodide.http import pyfetch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#download the dataset file and store as usedcars.csv
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
            
file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

await download(file_path, "usedcars.csv")
file_name="usedcars.csv"


df=pd.read_csv(file_name)



#1. Linear Regression and Multiple Linear Regression
#HERE WE WILL EXPLORE HOW DIFFERENT INDEPENDENT VAR AFFECT THE PRICE SO WE CAN CHOOSE WHICH ONE TO USE IN A MODEL

#Simple Linear Regression: how highway-mpg affects the price
lm=LinearRegression()
x=df[['highway-mpg']]
y=df[['price']]
lm.fit(x,y)
yhat=lm.predict(x)
print(lm.intercept_)
print(lm.coef_)
yhat

#Simple Linear Regression: how engine-size affects the price
lm1=LinearRegression()
x=df[['engine-size']]
y=df[['price']]
lm1.fit(x,y)
yhat1=lm1.predict(x)
print(lm1.intercept_)
print(lm1.coef_)
#yhat1=-7963.33890628+166.86001569*x

#Multiple Linear Regression: how horsepower,curb-weight, engine-size and highway-mpg affect the price
lmx=LinearRegression()
z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
#train the MLR model
lmx.fit(z,df[['price']])
yhatx=lmx.predict(z)
#yhatx cannot be manually calculated because Python gets stuck, we need sklearn lib?: -15806.62462633+53.49574423*df[['horsepower']]+4.70770099*df[['curb=weight']]+81.53026382*df[['engine-size']]+36.05748882*df[['highway-mpg']]
#print(yhatx)
print(lmx.intercept_)
print(lmx.coef_)

#Multiple Linear Regression: normlized-losses and highway-mpg affect the price
lm2=LinearRegression()
y=df[['price']]
z=df[['normalized-losses','highway-mpg']]
lm2.fit(z,y)
yhat2=lm2.predict(z)
print(lm2.intercept_)
print(lm2.coef_)


#2. Model Evaluation Using Visualization
# Regression plot to see is there correlation between highway-mpg and price, how strong is it and what is it's direction
#SLR model highway-mpg and price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.show()

#Regression plot to see is there correlation between peak-rpm and price
#SLR model peak-rpm and price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.show()

#Exploring correlation between peak-rpm + highway-mpg VS price
print(df[['peak-rpm','price']].corr())
print(df[['highway-mpg','price']].corr()) #highway-mpg clearly is more correlated with price

#Plotting the residuals between the predicted and actuall values 
#SLR model highway-mpg and price
sns.residplot(x=df['highway-mpg'],y=df['price'])
plt.show()
#We can see that residuals are not randomy dotted around the graph, but instead form a curve, which means our model is not appropriate
#Maybe we can consider non-linear model 


#MLR model cannot be visualised with regression or residual plot, so we use distribution plot
#we first calc Yhat for MLR model that shows correlation between muldtiple independent values (normalized-losses and highway-mpg) 
lm3=LinearRegression()
y=df[['price']]
z=df[['normalized-losses','highway-mpg']]
lm3.fit(z,y)
yhat3=lm3.predict(z)
#now we plot the distribution plot
ax1=sns.distplot(df['price'],hist=False,color='r',label='Actual value')
sns.distplot(yhat3,hist=False,color='b',label='Fitted value')
plt.title('Actual vs Fitted Values of price')
plt.xlabel('Price')
plt.ylabel('Number of samples')
plt.legend()
plt.show() #there is certain overlap between real and predicted line, but there is a lot of space for improvement
plt.close()


#3. Polynomial Regression and Pipelines
#We saw earlier that a SLR model using "highway-mpg" as the predictor variable did not do well. 
#Let's see if a polynomial model will do better

#3.1. Polynomial regression 1 indendent var
#defining method to plot the polynomial model
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)
    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')
    plt.show()
    plt.close()
#setting up variables for polynomial calculation
x = df['highway-mpg']
y = df['price']
# Here we use a polynomial of the 11rd order (cubic), we used 3 but id wasn't so good, so we increased the degree
#fitting the polynomial model
f = np.polyfit(x, y, 11)
#calculating the polynomial
p = np.poly1d(f) 
print(p) #prints the polynomial of 11th order function, using 1 independent var (highway_mpg): yhat=3.879e+0 - 1.491e+08 x + 2.551e+07 x(to2)  … -1.243e-08 x(to11) 
#plotting the polynomial we just calculated
PlotPolly(p, x, y, 'highway-mpg')
#how to get the predicted values?




#3.2. Polynimial regression with multiple indendent var - PIPELINE
#The analytical expression for Multivariate Polynomial function gets complicated
#For example, for second-order (degree=2) polynomial with two variables is: 
#yhat = a + b1x1 + b2x2 + b3x1x2 + b4x1x1 + b5x2x2
#so we need PIPELINE

#we create an object PolynomialFeatures, of degree 2
pr=PolynomialFeatures(degree=2)
#we choose independent variables
Z=df[['horsepower','curb-weight','engine-size','highway-mpg']]

 
#just for illustration
#we fit the model
Z_pr=pr.fit_transform(Z)
#now we can compare what we had in the beginning
print(Z.shape)#only 4 features (columns)
print(Z_pr.shape)#15 features, we "created" additional 11 features (colums)


# now we want to create pipeline, so we prepare input
input=[('polynomial',PolynomialFeatures(degree=2)),('scale',StandardScaler()),('model',LinearRegression())]
#call Pipeline constructor
pipe=Pipeline(input)
#fit Pipeline with data
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]


#Create a pipeline that standardizes the data, then produce a prediction using a linear regression model using the features Z and target y.
Input=[('scale',StandardScaler()),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]


#4. Measures for In-Sample Evaluation

#SIMPLE LINEAR REGRESSION EVALUATION
#Calculating R square for simple linear regression between highway-mpg and price
lm=LinearRegression()
x=df[['highway-mpg']]
y=df[['price']]
lm.fit(x,y)
print('SLR')
print('The R-square SLR is: ', lm.score(x, y)) # result of 0.49 means that 49% of the variation of the price is explained by this simple linear model "highway-mpg"
Yhat=lm.predict(x)
print('The output of the first four predicted values (SLR) is: ', Yhat[0:4])
#Calculating MSE for simple linear regression between highway-mpg and price
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value of SLR is: ', mse)

#MULTIPLE LINEAR REGRESSION EVALUATION
#Calculating R square for multiple linear regression between highway-mpg and price
#first we need to choose independent variables
Z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
#now we fit the model
lm2.fit(Z,y)
print('MLR')
#now we calculate R-square
print('The R-square of MLR is: ',lm2.score(Z,y)) #the result of 0.8 means that 80% of the price is explained by these 4 vars(horsepower, curb-weight, engine-size and highway-mpg)
#now we make prediction
Y_predict_multifit = lm2.predict(Z)
#now we calculate MSE
print('The mean square error of price and predicted value of MLR using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))


#POLYNOMIAL REGRESSION EVALUATION
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('PR')
print('The R-square value of 1 independent var polynomial is: ', r_squared)
print ('The MSE of 1 independent var polynomial is:', mean_squared_error(df['price'], p(x)))

#5. Prediction and Decision Making

#the model with the higher R-squared value is a better fit for the data
#the model with the smallest MSE value is a better fit for the data
#Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price we have: R-squared: 0.4965911884339176 and MSE: 31635042.944639888
#Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price we have R-squared: 0.8093562806577457 and MSE: 11980366.87072649
#Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price we have R-squared: 0.702376909243598 and MSE: 18703127.63915394
#CONCLUSION: the best fit is Multiple Linear Regression
