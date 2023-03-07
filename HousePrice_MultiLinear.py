#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# In[19]:


df=pd.read_csv("House_Price.csv")
print(df)


# In[21]:


#isnull() and any() functions from Pandas will determine if there are missing or invalid values
print(df.isnull().any())


# In[22]:


#prints the list of columns with missing values
print(df.dropna())


# In[23]:


#replacing missing values with mean or median
df.fillna(df.mean(), inplace=True) #This will replace all NaN values in the DataFrame with the mean value of the respective column.


# In[24]:


#independent variables
X_multi=df.drop(["price","airport","waterbody","bus_ter"],axis=1) #drop command will drop the price column and retreive all other
                                 #columns
                                #axis=1 specifies the columns


# In[25]:


print(X_multi)


# In[26]:


#creating dependent variable
y_multi=df["price"]


# In[27]:


print(y_multi)


# In[28]:


#splitting the data into test set and train set
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)


# In[29]:


#preprocessing the data by standardisation of test set and train set
scaler = StandardScaler() #assigning the StandardScaler function to scaler object
X_train = scaler.fit_transform(X_train) #train set transformation
X_test = scaler.transform(X_test) #test set transformation


# In[30]:


#training the train data using linear regression
model = LinearRegression() #assigning LinearRegression function to object model
model.fit(X_train, y_train) #fitting the training set


# In[31]:


#performing predictions on the test set
y_pred = model.predict(X_test)


# In[32]:


print(y_pred)


# # error calculation

# In[33]:


#mean squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)


# In[35]:


#r squared error
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)


# # visualization

# In[36]:


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)


# In[ ]:




