#!/usr/bin/env python
# coding: utf-8

# # Market Fluctuations and Predictability

# In[4]:


import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# In[5]:


df=pd.read_csv("C:\\Users\\MY Dell\\Downloads\\keshav\\flats.csv")


# In[16]:


df.head()


# In[18]:


df.columns



# In[21]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[129]:


# Label Encoding (for categorical features with limited categories)
categorical_features = ['property_name', 'society', 'facing', 'agePossession','age_bucket','price_per_sq_ft','price_per_sqft']
le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])


# In[130]:


# Label Encoding (for categorical features with limited categories)
categorical_features = ['link','price','rate','areaWithType','bedRoom','bathroom','balcony','additionalRoom','address','floorNum','nearbyLocations','description','furnishDetails','features','rating','property_id']
le = LabelEncoder()
for col in categorical_features:
    df[col] = le.fit_transform(df[col])


# In[131]:


df.info()


# In[132]:


df.isnull().sum()
df.dropna()


# In[133]:


# Create new features
df['price_per_sq_ft'] = df['price'] / df['areaWithType']


# In[134]:


# Exploratory Data Analysis (EDA)
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(df['price_per_sq_ft'], bins=30)
plt.title('Price per Square Foot Distribution')
plt.show()


# In[135]:


# 2. Location analysis
location_groups = df.groupby('address')  # Replace 'location' with your location column name
avg_price_sqft_loc = location_groups['price_per_sqft'].mean()


# In[136]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# In[137]:


df.info()


# In[138]:


df


# In[139]:


corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[140]:


# Prepare features and target variable
features = ['areaWithType', 'bedRoom', 'bathroom', 'balcony', 'agePossession', 'society', 'facing']
X = df[features]
y = df['price_per_sqft']


# In[141]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[142]:


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[143]:


# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f'Linear Regression MSE: {mse_lr}, MAE: {mae_lr}')


# In[144]:


# Train a random forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f'Random Forest MSE: {mse_rf}, MAE: {mae_rf}')


# In[145]:


# ARIMA model
model = ARIMA(df['price'], order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())


# In[146]:


#Forecast
forecast = model_fit.forecast(steps=10)
print(forecast)


# In[147]:


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(y_pred_lr, label='Predicted Prices - Linear Regression')
plt.plot(y_pred_rf, label='Predicted Prices - Random Forest')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('price_per_sqft')
plt.title('Actual vs Predicted Prices')
plt.show()


# In[ ]:




