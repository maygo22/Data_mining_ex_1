#!/usr/bin/env python
# coding: utf-8

# In[53]:


# ----------------  Read and load data ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV


test_path = r'C:\Users\migue\OneDrive\Desktop\test.csv'
train_path = r'C:\Users\migue\OneDrive\Desktop\train.csv'
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)

# Display the loaded data
print("Train Data:")
display(train_data.head(10))

print("\nTest Data:")
display(test_data.head(10))


# In[54]:


# ---------------- Data Preprocessing -------------------
X_train = train_data.drop(columns=['Y_defect_type'])
y_train = train_data['Y_defect_type']

# Check for missing values
missing_values = X_train.isna()

# Count missing values in each column
missing_counts = missing_values.sum()

print(missing_counts)

# Impute missing values with the median for each column
X_train_imputed = X_train.fillna(X_train.median())

log_transformed_data = np.log(X_train_imputed + 1)

display(log_transformed_data)

sc = StandardScaler()
X_train_sc=sc.fit_transform(log_transformed_data)

n_components = 10  # You can adjust this number
pca = PCA(n_components=n_components)

# Fit and transform the scaled data using PCA
X_train_pca = pca.fit_transform(X_train_sc)

#Create a new Pandas DataFrame with the principal components
columns = [f'PC{i}' for i in range(1, n_components + 1)]
X_train_pca = pd.DataFrame(data=X_train_pca, columns=columns)

explained_variance = pca.explained_variance_ratio_


# In[55]:


# ---------------- Build training model -------------------
n_splits = 5 
cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Use Random Forest
classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators as needed
scores = cross_val_score(classifier, X_train_pca, y_train, cv=cv_strategy, scoring='accuracy')
classifier.fit(X_train_pca, y_train)
print(scores)
print(f"Mean Accuracy: {scores.mean()}")
print(f"Standard Deviation: {scores.std()}")


# In[56]:


# ---------------- Predict on testing data -------------------
X_test = test_data.drop(columns=['Y_defect_type'])
y_test = test_data['Y_defect_type']

log_transformed_data = np.log(X_test + 1)

# Use the same StandardScaler as used for training data
X_test_sc = sc.transform(log_transformed_data)

# Apply PCA transformation
X_test_pca = pca.transform(X_test_sc)

# Create a new Pandas DataFrame with the principal components
columns = [f'PC{i}' for i in range(1, n_components + 1)]
X_test_pca = pd.DataFrame(data=X_test_pca, columns=columns)
y_pred = classifier.predict(X_test_pca)  
accuracy = accuracy_score(y_test, y_pred)


# In[57]:


# ---------------- Print your accuracy result -------------------
print(f"Testing Accuracy: {accuracy:.2f}")


# In[ ]:




