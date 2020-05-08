#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
from sklearn.metrics import mean_squared_error


# In[2]:


housing=pd.read_csv('Housing.csv')
housing


# In[3]:


housing.info()


# In[4]:


housing.describe()


# In[5]:


#converting yes to 1 and no to 0
housing['mainroad']=housing['mainroad'].map({'yes':1,'no':0})
housing['guestroom']=housing['guestroom'].map({'yes':1,'no':0})
housing['basement']=housing['basement'].map({'yes':1,'no':0})
housing['hotwaterheating']=housing['hotwaterheating'].map({'yes':1,'no':0})
housing['airconditioning']=housing['airconditioning'].map({'yes':1,'no':0})
housing['prefarea'] = housing['prefarea'].map({'yes': 1, 'no': 0})


# In[6]:


housing.head()


# In[7]:


status=pd.get_dummies(housing['furnishingstatus'])
status.head()


# In[8]:


#dropping furnished from the table
status=pd.get_dummies(housing['furnishingstatus'],drop_first=True)


# In[9]:


#adding semifurnished and unfurnished to the table
housing=pd.concat([housing,status],axis=1)


# In[10]:


housing.head()


# In[11]:


#dropping furnishing status
housing.drop(['furnishingstatus'],axis=1,inplace=True)


# In[12]:


housing.head()


# In[13]:


# Let us create the new metric and assign it to "areaperbedroom"
housing['areaperbedroom']=housing['area']/housing['bedrooms']


# In[14]:


# Metric:bathrooms per bedroom
housing['bbratio']=housing['bathrooms']/housing['bedrooms']


# In[15]:


housing.head()


# In[16]:


#defining a normalisation function 
def normalize(x):
    return((x-np.min(x))/(max(x)-min(x)))

# applying normalize ( ) to all columns 
housing=housing.apply(normalize)


# In[17]:


housing.head()


# In[18]:


housing.columns


# In[19]:


# Putting feature variable to X
X = housing[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'semi-furnished', 'unfurnished',
       'areaperbedroom', 'bbratio']]

# Putting response variable to y
y = housing['price']


# In[20]:


#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.7,random_state=100)


# In[21]:


import statsmodels.api as sm     # Importing statsmodels
X_train=sm.add_constant(X_train)  # Adding a constant column to our dataframe

# create a first fitted model
lm_1=sm.OLS(y_train,X_train).fit()


# In[22]:


#Let's see the summary of our first linear model
print(lm_1.summary())


# In[23]:


# UDF for calculating vif value
def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)


# In[24]:


# Calculating Vif value
vif_cal(input_data=housing, dependent_col="price")


# In[25]:


#correlation matrix


# In[26]:


# Let's see the correlation matrix 
plt.figure(figsize=(16,10))   # Size of the figure
sns.heatmap(housing.corr(),annot=True)


# In[27]:


# Dropping highly correlated variables and insignificant variables
X_train=X_train.drop('bbratio',1)


# In[28]:


# Create a second fitted model
lm_2=sm.OLS(y_train,X_train).fit()


# In[29]:


#Let's see the summary of our second linear model
print(lm_2.summary())


# In[30]:


# Calculating Vif value
vif_cal(input_data=housing.drop(["bbratio"], axis=1), dependent_col="price")


# In[31]:


# Dropping highly correlated variables and insignificant variables
X_train = X_train.drop('bedrooms',1)


# In[32]:


# Create a third fitted model
lm_3 = sm.OLS(y_train,X_train).fit()


# In[33]:


#Let's see the summary of our third linear model
print(lm_3.summary())


# In[34]:


# Calculating Vif value
vif_cal(input_data=housing.drop(["bedrooms","bbratio"], axis=1), dependent_col="price")


# In[35]:


# # Dropping highly correlated variables and insignificant variables
X_train = X_train.drop('areaperbedroom', 1)


# In[36]:


# Create a fourth fitted model
lm_4 = sm.OLS(y_train,X_train).fit()


# In[37]:


#Let's see the summary of our fourth linear model
print(lm_4.summary())


# In[38]:


# Calculating Vif value
vif_cal(input_data=housing.drop(["bedrooms","bbratio","areaperbedroom"], axis=1), dependent_col="price")


# In[39]:


# # Dropping highly correlated variables and insignificant variables
X_train = X_train.drop('semi-furnished', 1)


# In[40]:


# Create a fifth fitted model
lm_5 = sm.OLS(y_train,X_train).fit()


# In[41]:


#Let's see the summary of our fifth linear model
print(lm_5.summary())


# In[42]:


# Calculating Vif value
vif_cal(input_data=housing.drop(["bedrooms","bbratio","areaperbedroom","semi-furnished"], axis=1), dependent_col="price")


# In[43]:


# # Dropping highly correlated variables and insignificant variables
X_train = X_train.drop('basement', 1)


# In[44]:


# Create a sixth fitted model
lm_6 = sm.OLS(y_train,X_train).fit()


# In[45]:


#Let's see the summary of our sixth linear model
print(lm_6.summary())


# In[46]:


# Calculating Vif value
vif_cal(input_data=housing.drop(["bedrooms","bbratio","areaperbedroom","semi-furnished","basement"], axis=1), dependent_col="price")


# In[47]:


# Adding  constant variable to test dataframe
X_test_m6 = sm.add_constant(X_test)


# In[48]:


# Creating X_test_m6 dataframe by dropping variables from X_test_m6
X_test_m6 = X_test_m6.drop(["bedrooms","bbratio","areaperbedroom","semi-furnished","basement"], axis=1)


# In[49]:


# Making predictions
y_pred_m6 = lm_6.predict(X_test_m6)


# In[59]:


# Actual vs Predicted
c = [i for i in range(1,165,1)]
fig = plt.figure()
plt.plot(y_test, color="blue", linewidth=2.5, linestyle="-")     #Plotting Actual
plt.plot(y_pred_m6, color="red",  linewidth=2.5, linestyle="-")  #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Housing Price', fontsize=16)                       # Y-label


# In[51]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred_m6)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# In[55]:


# Error terms
fig = plt.figure()
c = [i for i in range(1,165,1)]
plt.plot(c,y_test-y_pred_m6, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label


# In[53]:


# Plotting the error terms to understand the distribution.
fig = plt.figure()
sns.distplot((y_test-y_pred_m6),bins=50)
fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)                          # Y-label


# In[54]:


import numpy as np
from sklearn import metrics
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, y_pred_m6)))


# In[ ]:




