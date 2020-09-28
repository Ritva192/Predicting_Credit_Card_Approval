#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[21]:


#Loading the dataset
credit_data = pd.read_csv('C:\credit-approval_csv.csv')
#Looking at the dataset
credit_data.head()
#
credit_description = credit_data.describe()
print(credit_description)


# In[22]:


credit_info = credit_data.info()
print(credit_info)


# In[23]:


# Replace "?" with NaN
credit_data.replace('?', np.NaN, inplace = True)
#Replacing the misssing value by mean value
credit_data.fillna(credit_data.mean(), inplace=True)
# Convert Age to numeric
credit_data["Age"] = pd.to_numeric(credit_data["Age"])
print(credit_data.isnull().sum())


# In[6]:


for col in credit_data.columns:
    # Check if the column is of object type
    if credit_data[col].dtypes == 'object':
        # Impute with the most frequent value
        credit_data = credit_data.fillna(credit_data[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(credit_data.isnull().sum())


# In[7]:


# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le=LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in credit_data.columns.values:
    # Compare if the dtype is object
    if credit_data[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
        credit_data[col]=le.fit_transform(credit_data[col])


# In[14]:


# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop the features 11 and 13 and convert the DataFrame to a NumPy array
credit_data = credit_data.drop(['DriversLicense', 'ZipCode'], axis=1)
credit_data = credit_data.values


# Segregate features and labels into separate variables
X,y = credit_data[:,0:13] , credit_data[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X
                                                    ,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)


# In[15]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)


# In[16]:


# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train,y_train)


# In[17]:


# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test,y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test,y_pred)


# In[18]:


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001 ,0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are the corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)


# In[19]:


# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit grid_model to the data
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))


# In[ ]:




