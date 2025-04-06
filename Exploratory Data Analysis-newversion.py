#Exploratory Data Analysis

import pandas as pd
import numpy as np
import piplite
await piplite.install('seaborn')
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy import stats
from pyodide.http import pyfetch
from scipy import stats

#download the dataset and load it into pdf
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

file_path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
await download(file_path, "usedcars.csv")
file_name="usedcars.csv"
df = pd.read_csv(file_name, header=0)

#check columns datatypes
df.dtypes


#1. ANALYZING CONTINUOUS VARIABLES (DATATYPE FLOAT OR INT) - corr
#regression plots and corr table
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
numeric_df.corr()

#scatterplot the correlation between the engine-size and the target var - price
#sns.regplot(x="engine-size", y="price", data=df)
#plt.show()#the plot shows positive linear relation
#we can also calculate the correlation
df[["engine-size", "price"]].corr() #strong positive relation, can be a good predictor of the price

#scatterplot the correlation between the highway-mpg and the price
#sns.regplot(x="highway-mpg", y="price", data=df)
#plt.show()#the plot shows negative linear relation
#calculate the correlation
df[['highway-mpg', 'price']].corr()#strong negative relation, can be a good predictor of the price

#scatterplot the correlation between the peak-rpm and the price
#sns.regplot(x="peak-rpm", y="price", data=df)
#plt.show()#the plot shows weak or no linear relation
#calculate the correlation
df[['peak-rpm', 'price']].corr()#weak or no linear relation, may not be a good predictor of the price


df[["stroke","price"]].corr()#weak or no linear relation, may not be a good predictor of the price
#sns.regplot(x="stroke", y="price", data=df)
#plt.show()#the plot shows weak or no linear relation
#...etc for all continuous vars
#Conclusion - 'bore','compression-ratio','horsepower','wheel-base','length', 'width', 'curb-weight','engine-size',city-mpg', 'highway-mpg' have a high correlation with 'price', we should analyse them further



#2. ANALYZING CATEGORICAL VARIABLES (INT OR OBJECT DATATYPE) - boxplot
#box plots
#relationship between "body-style" and "price"
#sns.boxplot(x="body-style", y="price", data=df)
#plt.show()#the distributions of price between the different body-style categories have a significant overlap, so body-style would not be a good predictor of price
#let's try other categorical variables, such as engine-location, and drive wheels
#sns.boxplot(x="engine-location", y="price", data=df)
#plt.show()#the distributions of price between the front and rear engine location is clear, so engine-location may be a good predictor of price
#sns.boxplot(x="drive-wheels", y="price", data=df)
#plt.show()#the distributions of price between rwd vs fwd and 4wd  is clear but fwd and 4fd has a significant overlap, so drive-wheels may or may not be a good predictor of price
#...etc for all categorical vars
#Conclusion - engine-location and drive-wheels may be good predictors

#3. ANALYZING CONTINUOUS AND CATEGORICAL VARIABLES OF INTEREST - DESCRIPTIVE STATISTICAL ANALYSIS 
df.describe()
#The default setting of "describe" skips variables of type object, but we will include it
#df.describe(include=['object'])
#understanding how many units of each value in a column we have
#method "value_counts" only works on pandas series, so we use one bracket [], not two [[]]
df['engine-location'].value_counts() #not a good predictor of the price after all, because we have very little observations with engine-location in the rear
df['drive-wheels'].value_counts() #maybe a bit better as a predictor of the price? 
#... etc for all vars of interest, CONTINUOUS and CATEGORICAL 
#conclusion drive-wheels is the only categorical var we will take into account (why not make I do not know)

'''
#4. ANALYZING CATEGORICAL VARIABLES - GROUPING/PIVOT
#We can select the columns of interest and keep them in a separate df
temp_df = df[['drive-wheels','body-style','price']]
#we can group by one column and see which drive-wheels is related to the highest avg price
df_grouped_one_col = temp_df.groupby(['drive-wheels'], as_index=False).agg({'price': 'mean'})
df_grouped_one_col
#or we can group by 2 columns and see which drive-wheels/body-style combination is related to the highest avg price
df_grouped_two_col = temp_df.groupby(['drive-wheels','body-style'], as_index=False).agg({'price': 'mean'})
df_grouped_two_col
#df_grouped_two_col is easier to visualize when it is made into a pivot table
grouped_pivot = df_grouped_two_col.pivot(index='drive-wheels',columns='body-style')
grouped_pivot
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0, those are body-styles that do not have certain drive-wheel options
grouped_pivot
#lets visualise this pivot table on a heat map
plt.pcolor(grouped_pivot,cmap='RdBu')
plt.colorbar()
#prettify the heat map
plt.xlabel('body-style') #this will affect every other plt.show() in this notebook
plt.ylabel('drive-wheels') #this will affect every other plt.show() in this notebook
#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index
#set xicks and yticks labels
plt.xticks([1, 2, 3, 4, 5], row_labels) #this will affect every other plt.show() in this notebook
plt.yticks([1, 2, 3], col_labels) #this will affect every other plt.show() in this notebook
#display heat map
#plt.show()
'''

'''
#5. ANALYZING CONTINUOUS VARIABLES (DATATYPE FLOAT OR INT)CORRELATION AND CAUSATION
#Correlation: a measure of the extent of interdependence between variables.
#Causation: the relationship between cause and effect between two variables.
#Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient for wheel-base is", pearson_coef, " with a P-value of P =", p_value)  #Since the p-value is < 0.001, the correlation is statistically significant, although the linear relationship isn't extremely strong (~0.585).
#Pearson Correlation Coefficient and P-value for other NUMERICAL (some of them are CATEGORICAL VARS) cols vs price
pearson_coef, p_value = stats.pearsonr(df['symboling'], df['price'])
print("The Pearson Correlation Coefficient for symboling is", pearson_coef, " with a P-value of P = ", p_value) #not important
pearson_coef, p_value = stats.pearsonr(df['normalized-losses'], df['price'])
print("The Pearson Correlation Coefficient for normalized-losses is", pearson_coef, " with a P-value of P = ", p_value) #not important
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient for horsepower is", pearson_coef, " with a P-value of P = ", p_value) #important
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient for length is", pearson_coef, " with a P-value of P = ", p_value) #imortant
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient for width is", pearson_coef, " with a P-value of P = ", p_value) #important
pearson_coef, p_value = stats.pearsonr(df['height'], df['price'])
print("The Pearson Correlation Coefficient for height is", pearson_coef, " with a P-value of P = ", p_value) #not important
#etc... for all NUMERICAL (some of them are CATEGORICAL VARS) columns
'''

'''
#CONCLUSION
We have narrowed it down to the following variables:

Continuous numerical variables:
Length
Width
Curb-weight
Engine-size
Horsepower
City-mpg
Highway-mpg
Wheel-base
Bore

Categorical variables:
Drive-wheels
'''

