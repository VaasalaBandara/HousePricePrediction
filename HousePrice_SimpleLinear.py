#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# In[2]:


#importing the csv file as a dataframe using pandas
df=pd.read_csv("House_Price.csv")
print(df)


# In[3]:


#assigning the independent variable
X=df[["room_num"]]#x variable should be a two dimensional array


# In[4]:


#assigning the dependent variable
y=df["price"]


# In[5]:


#splitting the dataset into test set and train set
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)


# In[6]:


linoB=LinearRegression() #the object for the linear regression function


# In[8]:


#fitting the model
linoB.fit(X_train,y_train)


# In[9]:


#see untercepts and coefficients
print(linoB.intercept_,linoB.coef_)#underscores mean that they are attributes of linear regression model


# In[10]:


#predicting the values of the test set
y_pred = linoB.predict(X_test)


# In[11]:


#the predicted values
print(y_pred)


# # error calculation

# evaluation metrics for model

# In[16]:


#mean squared error
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
print("mean square errror is:",mse)


# In[17]:


#root mean squared error
import numpy as np
rmse=np.sqrt(mse)
print("root mean squared error is:",rmse)


# In[14]:


#r squared error
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("r squared error is:",r2)


# # visualization

# In[19]:


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)


# In[ ]:




