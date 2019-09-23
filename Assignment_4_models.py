#!/usr/bin/env python
# coding: utf-8

# In[92]:


# Importing necessary libraries
import sklearn
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns


# In[93]:


conc_df = pd.read_csv("concrete.csv")


# In[94]:


conc_df.describe()


# In[95]:


X = conc_df.drop("strength", axis =1)
y = conc_df["strength"]


# In[96]:


conc_df.head()


# In[97]:


X.head()


# In[98]:


y.head()


# In[99]:


X.shape


# In[100]:


y.shape


# In[101]:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

sc_x  = StandardScaler()
sc_y  = StandardScaler()

X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y[:, np.newaxis]).flatten()


# In[102]


# In[103]:


# Splitting the dataset randomly

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


# In[104]:


X_train.shape


# In[105]:


y_train.shape


# In[106]:


X_test.shape


# In[107]:


y_train.shape


# In[108]:


type(X_train)
type(X_test)


# In[109]:


from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X_train,y_train)


# In[110]:


slr.intercept_


# In[111]:


y_train_pred = slr.predict(X_train)
y_test_pred  = slr.predict(X_test)


# In[112]:


y_test_pred[0]


# In[113]:


slr.coef_


# In[114]:


res_plot = plt.scatter(y_train_pred,  y_train_pred - y_train,
                       c='steelblue', 
                       marker='o', edgecolor='white',
                       label='Training data')
res_plot = plt.scatter(y_test_pred,  y_test_pred - y_test,
                       c='limegreen', marker='s', edgecolor='white',
                       label='Test data')
res_plot = plt.xlabel('Predicted values')
res_plot = plt.ylabel('Residuals')
res_plot = plt.legend(loc='upper left')
res_plot = plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
res_plot = plt.xlim([-10, 50])

plt.show()


# In[115]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[116]:


MSE_train = mean_squared_error(y_train, y_train_pred)
MSE_test  = mean_squared_error(y_test, y_test_pred)

R_squared_train = r2_score(y_train, y_train_pred)
R_squared_test  = r2_score(y_test, y_test_pred) 


# In[117]:


print("MSE_train: ", MSE_train, "\n", "MSE_test: ", MSE_test, "\n", 
      "R_squared_train: ", R_squared_train, "\n", "R_squared_test: ", R_squared_test, "\n")


# In[158]:


from sklearn.linear_model import Ridge

MSE_train = []
MSE_test  = []
R_sq_train = []
R_sq_test  = []


for alpha in [0,0.01,0.5,1.0,1.5,2.0,3.0,4.0,5.0,8.0,10.0,15.0,100]:
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train,y_train)
    
    y_train_pred = ridge.predict(X_train)
    y_test_pred  = ridge.predict(X_test)
    
    MSE_train.append(mean_squared_error(y_train, y_train_pred))
    MSE_test.append(mean_squared_error(y_test, y_test_pred))
    R_sq_train.append(r2_score(y_train, y_train_pred))
    R_sq_test.append(r2_score(y_test, y_test_pred))
    


# In[159]:


MSE = plt.plot(MSE_train,"g",label = "training set MSE")
MSE = plt.plot(MSE_test,"r", label = "test set MSE")
MSE = plt.legend()
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.show()


# In[160]:


R_sq = plt.plot(R_sq_train,"g",label = "training set R_sq")
R_sq = plt.plot(R_sq_test,"r", label = "test set R_sq")
R_sq = plt.legend()
plt.xlabel("alpha")
plt.ylabel("R_sq")
plt.show()


# In[161]:


MSE_train


# In[162]:


MSE_test


# In[163]:


R_sq_train


# In[164]:


R_sq_test


# In[191]:


# fitting the model with alpha as 1
ridge = Ridge(alpha = 1)
ridge.fit(X_train,y_train)

y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)


# In[192]:


ridge.intercept_


# In[193]:


ridge.coef_


# In[194]:


res_plot = plt.scatter(y_train_pred,  y_train_pred - y_train,
                       c='steelblue', 
                       marker='o', edgecolor='white',
                       label='Training data')
res_plot = plt.scatter(y_test_pred,  y_test_pred - y_test,
                       c='limegreen', marker='s', edgecolor='white',
                       label='Test data')
res_plot = plt.xlabel('Predicted values')
res_plot = plt.ylabel('Residuals')
res_plot = plt.legend(loc='upper left')
res_plot = plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
res_plot = plt.xlim([-10, 50])

plt.show()


# In[195]:


MSE_train = mean_squared_error(y_train, y_train_pred)
MSE_test  = mean_squared_error(y_test, y_test_pred)

R_squared_train = r2_score(y_train, y_train_pred)
R_squared_test  = r2_score(y_test, y_test_pred) 


# In[196]:


print("Ridge","\n\n","MSE_train: ", MSE_train, "\n", "MSE_test: ", MSE_test, "\n", 
      "R_squared_train: ", R_squared_train, "\n", "R_squared_test: ", R_squared_test, "\n")


# In[186]:


from sklearn.linear_model import Lasso

MSE_train = []
MSE_test  = []
R_sq_train = []
R_sq_test  = []


for alpha in [0.001,0.01,0.03,0.5,1.0,1.5,2.0]:
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train,y_train)
    
    y_train_pred = lasso.predict(X_train)
    y_test_pred  = lasso.predict(X_test)
    
    MSE_train.append(mean_squared_error(y_train, y_train_pred))
    MSE_test.append(mean_squared_error(y_test, y_test_pred))
    R_sq_train.append(r2_score(y_train, y_train_pred))
    R_sq_test.append(r2_score(y_test, y_test_pred))


# In[187]:


MSE_train


# In[188]:


MSE_test


# In[189]:


R_sq_test


# In[190]:


R_sq_train


# In[197]:


# fitting the lasso model with alpha as 0.01
lasso = Lasso(alpha = 0.01)
lasso.fit(X_train,y_train)

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)


# In[198]:


lasso.intercept_


# In[199]:


lasso.coef_


# In[200]:


res_plot = plt.scatter(y_train_pred,  y_train_pred - y_train,
                       c='steelblue', 
                       marker='o', edgecolor='white',
                       label='Training data')
res_plot = plt.scatter(y_test_pred,  y_test_pred - y_test,
                       c='limegreen', marker='s', edgecolor='white',
                       label='Test data')
res_plot = plt.xlabel('Predicted values')
res_plot = plt.ylabel('Residuals')
res_plot = plt.legend(loc='upper left')
res_plot = plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
res_plot = plt.xlim([-10, 50])

plt.show()


# In[201]:


MSE_train = mean_squared_error(y_train, y_train_pred)
MSE_test  = mean_squared_error(y_test, y_test_pred)

R_squared_train = r2_score(y_train, y_train_pred)
R_squared_test  = r2_score(y_test, y_test_pred) 


# In[202]:


print("Lasso","\n\n","MSE_train: ", MSE_train, "\n", "MSE_test: ", MSE_test, "\n", 
      "R_squared_train: ", R_squared_train, "\n", "R_squared_test: ", R_squared_test, "\n")





print("My name is Rakesh Reddy Mudhireddy")
print("My NetID is: rmudhi2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# End
