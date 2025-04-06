import pandas as pd
import matplotlib.pylab as plt
#1
from pyodide.http import pyfetch 
#2
import numpy as np 
#6


#1. DOWNLOADING CSV
'''
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())
file_path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
await download(file_path, "usedcars.csv")
file_name="usedcars.csv"
#define headers for the df
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
'''

#read csv to df
df = pd.read_csv('auto.csv', names = headers)



#2. HANDLING MISSING DATA
# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
#find the null values across all cols in the df
df.isnull().sum() #prints out a table with col names and number of null values
#calc mean of normalized-losses column and use it to replace null values
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
df["normalized-losses"]=df["normalized-losses"].replace(np.nan,avg_norm_loss)
#find the most frequent value of num-of-doors column and use it to replace null values
most_frequent=df["num-of-doors"].value_counts().idxmax()
df["num-of-doors"]=df["num-of-doors"].replace(np.nan,most_frequent)
#calc mean of bore column and use it to replace null values
avg_bore = df["bore"].astype("float").mean(axis=0)
df["bore"]=df["bore"].replace(np.nan,avg_bore)
#calc mean of stroke column and use it to replace null values
avg_stroke = df["stroke"].astype("float").mean(axis=0)
df["stroke"]=df["stroke"].replace(np.nan,avg_stroke)
#calc mean of horsepower column and use it to replace null values
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
df["horsepower"]=df["horsepower"].replace(np.nan,avg_horsepower)
#calc mean of peak-rpm column and use it to replace null values
avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
df["peak-rpm"]=df["peak-rpm"].replace(np.nan,avg_peak_rpm)
# delete rows with null data in price column, since price is target var no prediction can be made without it
df=df.dropna(subset=['price'],axis=0)


#3. CORRECTING DATA FORMAT
#print out the column names and datatypes
df.dtypes
#bore, stroke, peak-rpm and price should be float, not object 
#USE DOUBLE [[]]
df[['bore', 'stroke', 'peak-rpm', 'price']]=df[['bore', 'stroke', 'peak-rpm', 'price']].astype('float')
#normalized-losses and horsepower should be int, not object
df[['normalized-losses','horsepower']]=df[['normalized-losses','horsepower']].astype(int)


#4. DATA STANDARDIZATION
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
df['highway-L/100km'] = 235/df["highway-mpg"]


#5. DATA NORMALIZATION
#Why normalization? In order to transform values of several variables into a similar range
#To demonstrate normalization, we will scale the columns length, width and height, using (original value)/(maximum value)
df['width'] = df['width']/df['width'].max()
df['length'] = df['length']/df['length'].max()
df['height'] = df['height']/df['height'].max()


#6. BINNING (GROUPING)
#Why binning? To transform continuous numerical variables into discrete categorical 'bins' for grouped analysis

df["horsepower"]=df["horsepower"].astype(int)

#plotting the histogram to see the distribution, not mandatory
%matplotlib inline
import matplotlib.pyplot as plt
plt.hist(df["horsepower"])
# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()

#create bins 
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
#create bins names
group_names = ["Low", "Medium", "High"]
#add bins column to dataframe
df["horsepower-bins"]=pd.cut(df["horsepower"],bins,labels=group_names,include_lowest=True)

#see the number of vehicles in each bin - not mandatory
df["horsepower-bins"].value_counts()

#visualise the bins - not mandatory
%matplotlib inline
import matplotlib.pyplot as plt
plt.bar(group_names, df["horsepower-bins"].value_counts())
# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
#plt.show()


#7. __
#Why indicator variables? So we can use categorical variables for further analysis 
#create dummy columns from column fuel-type
dummy_cols=pd.get_dummies(df['fuel-type'])
#concatenate dummy columns to the df
df=pd.concat([df,dummy_cols],axis=1)
#rename dummy cols so we know where they came from
df=df.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'})
#drop column fuel-type
df=df.drop('fuel-type',axis=1)

#create dummy columns from column aspiration
dummy_cols=pd.get_dummies(df['aspiration'])
df=pd.concat([df,dummy_cols],axis=1)
df=df.rename(columns={'std':'aspiration-std','turbo':'aspiration-turbo'})
df=df.drop('aspiration',axis=1)

