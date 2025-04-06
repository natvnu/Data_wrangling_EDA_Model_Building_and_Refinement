# Data_wrangling_EDA_Model_Building_and_Refinement

The repository contains:
1. DataWrangling-UsedCars.py - Importing dataset into df, handling missing data, indicator vars, correcting data format, data standardization and normalizaton, binning.
2. Exploratory Data Analysis-newversion.py - Exploring features to predict price of car, analyzing patterns grouping of data and creating pivot tables. Identifying the effect of independent attributes on price of cars using seaborn plots.
3. MODEL DEVELOPMENT.py - Simple and Multivar Linear Regression, Polynomial Transformation, Pipelines
4. .py - Model evaluation and refinement - Cross Validation, train-test split, determining point of overfitting, Ridge Regression, GridSearch
5. .py


7. 4 csv files (auto.csv (unprocessed dataset), usedcars.csv, laptops.csv and insurance.csv)

Dataset Sources: 
Csv files: auto.csv (unprocessed dataset), usedcars.csv, laptops.csv and insurance.csv, acquired from IBM Data Analysis with Python course (https://www.coursera.org/learn/data-analysis-with-python?specialization=ibm-data-science)

Technologies Used: Python, Pandas, seaborn, numpy, piplite, matplotlib, sklearn, pyodide 

Installation: copy and run the code in Jupyter Notebooks or other Python editor of choice. If the csv files are not downloaded once py files is executed, then store them in the same folder.

Example of results:

![Data_wrangling_bins](https://github.com/natvnu/Data_wrangling_EDA_Model_Building_and_Refinement/blob/main/bins.png?raw=true)

![EDA_regplot_used_cars](https://github.com/natvnu/Data_wrangling_EDA_Model_Building_and_Refinement/blob/main/regplot.png?raw=true)

![EDA_boxplot_used_cars](https://github.com/natvnu/Data_wrangling_EDA_Model_Building_and_Refinement/blob/main/boxplot.png?raw=true)

![EDA_heatmap_used_cars](https://github.com/natvnu/Data_wrangling_EDA_Model_Building_and_Refinement/blob/main/heatmap.png?raw=true)

![Model_building_distribution_plot](https://raw.githubusercontent.com/natvnu/Data_wrangling_EDA_Model_Building_and_Refinement/9088d19e6233ab9f15e590ee0aa828d3a71c0db1/Actual%20vs%20fitted%20target%20var.png)

![Model_building_Evaluation](https://raw.githubusercontent.com/natvnu/Data_wrangling_EDA_Model_Building_and_Refinement/9088d19e6233ab9f15e590ee0aa828d3a71c0db1/R2.png)






