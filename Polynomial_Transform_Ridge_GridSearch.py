
%pip install seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
%matplotlib inline

#This function will download the dataset into your browser 
from pyodide.http import pyfetch
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
await download(path, "laptops.csv")
file_name="laptops.csv"
df = pd.read_csv(file_name, header=0)
df.head()

#data wrangling
df=df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)
df_numeric=df.select_dtypes(include=['float64','int64'])
df_numeric.corr()

#build a model using Ridge Regression with polynomialy transformed x features
x=df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU', 'Category']]
y=df[['Price']]
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.5,random_state=1)
#polynomialy transform x - the set of features
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)
df_x_train_pr = pd.DataFrame(x_train_pr)
df_x_train_pr.shape #shows that no of features changed from 7 to 36 through polinomial transformation
#ridge regression
alpha_param=np.arange(1,5,0.1)
scores=[]
#calculate R2 for the range of alpha_param
for a in alpha_param:
    RR=Ridge(alpha=a)
    RR.fit(x_train_pr, y_train)
    scores.append(RR.score(x_test_pr,y_test))
#Plot the R2 values with respect to the value of alpha 
plt.figure(figsize=(15, 6))
plt.plot(alpha_param,scores)
plt.title('Relation between R2 score and alpha')
plt.xlabel('alpha')
plt.xticks(np.arange(1,5,0.1))
plt.ylabel('R2 scores')
plt.show()#around alpha 1.75 R2 starts to slow down with rise

#with the fact that around 1.75 R2 starts to slow down in mind, we will explore alpha around that value
#GridSearch incorporates the loop we created above, performs cross validation and finds the best alpha automatically
alpha_parameters=[{'alpha':[1.5,1.6,1.7,1.8,1.9,2]}]
#create RidgeRegression object
RR=Ridge()
#create GridSearchCV object and pass RegressionObject, parameters and no of folds
GS = GridSearchCV(RR, alpha_parameters, cv=4)
#fit the GridSearch model with train data
GS.fit(x_train_pr,y_train) #if we passed x_train results would be different and best R2 lower
#calc the best estimator (Ridge object that contains the optimal alpha)
best_alpha=GS.best_estimator_
all_scores=GS.cv_results_
best_score=best_alpha.score(x_test_pr,y_test)
print('Optimal value for alpha is:', best_alpha) #Ridge(alpha=2)
print('Best R2 score for the above alpha is: ', best_score) #0.5192153919521125

x_train_pr.shape
