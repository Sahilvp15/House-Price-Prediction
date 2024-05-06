#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
warnings.filterwarnings('ignore')


# Part 1 Predicting Professor Salary
# Load the dataset
# Perform EDA on this dataset
# Remove outliers
# Separate numerical features from categorical features
# Build a two-factors model to predict the Salary with both YrsSincePhd and YrsOfService as its correlation is higher
# Check if the model pass cross validation
# Use one-hot encoding to include the Rank, Sex and Discipline along with the above numerical variable to build a second model.
# Comment on whether the model improve or not after adding the categorical variables in terms of model performance as well as validation

# In[2]:


housing = pd.read_csv("opa_properties_public.csv")
housing.head()


# In[3]:


housing.describe()


# In[4]:


housing.columns


# In[5]:


drop_column_list = ['the_geom','type_heater', 'unfinished', 'utility', 'view_type', 'topography','suffix','fuel','exterior_condition','exempt_land', 'exempt_building','assessment_date','beginning_point', 'book_and_page', 'category_code_description', 'cross_reference', 'house_number','location', 'mailing_address_1', 'mailing_address_2', 'mailing_care_of', 'mailing_city_state', 'mailing_street', 'market_value_date','the_geom_webmercator','other_building','owner_1', 'owner_2', 'parcel_number', 'recording_date', 'sale_date', 'registry_number', 'unit', 'objectid','building_code', 'census_tract', 'date_exterior_condition', 'year_built_estimate', 'house_extension', 'mailing_zip', 'sewer', 'site_type','state_code',                    'street_designation', 'street_name', 'street_direction',       'geographic_ward']
data = housing.drop(drop_column_list, axis = 1)


# In[6]:


data.columns


# In[7]:


data.number_of_bathrooms


# In[8]:


data = data.dropna(subset=['number_of_bathrooms'])


# In[9]:


bathroom_counts = data['number_of_bathrooms'].value_counts()
print(bathroom_counts)


# After looking at the dataset I noticed that there are NAN values inside the market value, as this is a very oimportant festure, we will replsce the NANs with the median

# In[10]:


data = data.dropna(subset=['market_value'])


# In[11]:


data.isnull().any()


# In[12]:


data['market_value'] = data['market_value'].fillna(data['market_value'].mean())
data['total_area'] = data['total_area'].fillna(data['total_area'].mean())



# In[13]:


sns.boxplot(x=data['market_value'])


# In[14]:


#Removing YrsOfService outliers
data = data[ data.market_value < 200000]
sns.boxplot(x=data['market_value'])


# In[15]:


#Removing YrsOfService outliers
data = data[ data.total_area < 2300]
sns.boxplot(x=data['total_area'])


# In[16]:


data


# # Linear regression 

# In[17]:


Xarray = data['market_value'].values
Yarray = data['total_area'].values


# In[18]:


X = Xarray.reshape(-1, 1)
Y = Yarray.reshape(-1, 1)


# In[19]:


model1 = LinearRegression()
model1.fit(X, Y)


# In[20]:


Y_pred = model1.predict(X)


# In[21]:


Y_pred 


# In[22]:


plt.scatter(X, Y,  color='gray')
plt.plot(X, Y_pred, color='red', linewidth=2)
plt.show()


# In[23]:


#Splitting Data intro training and testing set
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[24]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(0.8 * data.shape[0])
print(0.2 * data.shape[0])


# In[25]:


#Splitting Data intro training and testing set
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[26]:


model2 = LinearRegression()
model2.fit(X_train, Y_train)


# In[27]:


Y_pred = model2.predict(X_test)
plt.scatter(X_test, Y_test,  color='gray')
plt.scatter(X_test, Y_pred, color='red', linewidth=2)
plt.show()


# In[28]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R-squared:', metrics.r2_score(Y_test, Y_pred)) # r^2 value isnt up to par becuase of how larger the data is 


# ## Linear regression Evalulation Metrics 

# In[29]:


predictions = model2.predict(X_test)


# In[30]:


# Assuming predictions are continuous values
threshold = 0.5  
predictions_binary = [1 if p >= threshold else 0 for p in predictions]

# Assuming Y_test is continuous
Y_test_categorical = [1 if y >= threshold else 0 for y in Y_test]

print(classification_report(Y_test_categorical, predictions_binary))
print(accuracy_score(Y_test_categorical, predictions_binary))


# ## Building a two factor model to predict the price with both the number of bathrooms and total area as its coloration is higher 

# In[31]:


X = data[['total_area','number_of_bathrooms']].values.reshape(-1, 2)
Y = data['market_value'].values.reshape(-1, 1)
 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


# In[32]:


model2f = LinearRegression()
model2f.fit(X_train, Y_train)
Y_pred = model2f.predict(X_test)
print(model2f.coef_)
print(model2f.intercept_)


# In[33]:


plt.scatter(Y_test, Y_pred, color='red', linewidth=2)
plt.show()


# In[34]:


print('R-squared:', metrics.r2_score(Y_test, Y_pred))


# In[35]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('R-squared:', metrics.r2_score(Y_test, Y_pred)) # r^2 value isnt up to par becuase of how larger the data is 


# # Decision Tree 

# In[36]:


from sklearn.tree import DecisionTreeRegressor

# Create and fit the DecisionTreeRegressor
tree_model = DecisionTreeRegressor(random_state=101)
tree_model.fit(X_train, Y_train)


# In[37]:


# Predictions
Y_pred = tree_model.predict(X_test)

# Evaluate the model
threshold = 0.5  
predictions_binary = [1 if p >= threshold else 0 for p in predictions]

# Assuming Y_test is continuous
Y_test_categorical_Decision_Tree = [1 if y >= threshold else 0 for y in Y_test]


# ## Decision Tree Evalulation Metrics 

# In[38]:


print(classification_report(Y_test_categorical_Decision_Tree, predictions_binary))
print(accuracy_score(Y_test_categorical_Decision_Tree, predictions_binary))


# In[39]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_categorical_Decision_Tree, predictions_binary))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_categorical_Decision_Tree, predictions_binary))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_categorical_Decision_Tree, predictions_binary)))
print('R-squared:', metrics.r2_score(Y_test_categorical_Decision_Tree, predictions_binary)) # r^2 value isnt up to par becuase of how larger the data is 


# # Random Forest 

# In[40]:


from sklearn.ensemble import RandomForestRegressor

# Create and fit the Random Forest
rmd_clf = RandomForestRegressor(random_state=101)
rmd_clf.fit(X_train, Y_train)

# Predictions
Y_pred = rmd_clf.predict(X_test)

# Evaluate the model
threshold = 0.5  
predictions_binary = [1 if p >= threshold else 0 for p in predictions]

# Assuming Y_test is continuous
Y_test_categorical_Randomforest = [1 if y >= threshold else 0 for y in Y_test]


# ## Random Forest Evalulation Metrics 

# In[41]:


print(classification_report(Y_test_categorical_Randomforest, predictions_binary))
print(accuracy_score(Y_test_categorical_Randomforest, predictions_binary))


# In[42]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_categorical_Randomforest, predictions_binary))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_categorical_Randomforest, predictions_binary))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_categorical_Randomforest, predictions_binary)))
print('R-squared:', metrics.r2_score(Y_test_categorical_Randomforest, predictions_binary))


# # Gradient Boosting

# In[43]:


from sklearn.ensemble import GradientBoostingRegressor

gb_clf = GradientBoostingRegressor(learning_rate=0.5, random_state=100)
gb_clf.fit(X_train, Y_train)

Y_pred = gb_clf.predict(X_test)

threshold = 0.5  
predictions_binary = [1 if p >= threshold else 0 for p in predictions]

# Assuming Y_test is continuous
Y_test_categorical_Gb = [1 if y >= threshold else 0 for y in Y_test]


# ## Gradient Boosting Evalulation Metrics

# In[44]:


print(classification_report(Y_test_categorical_Gb, predictions_binary))
print(accuracy_score(Y_test_categorical_Gb, predictions_binary))


# In[45]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test_categorical_Gb, predictions_binary))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test_categorical_Gb, predictions_binary))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test_categorical_Gb, predictions_binary)))
print('R-squared:', metrics.r2_score(Y_test_categorical_Gb, predictions_binary))

