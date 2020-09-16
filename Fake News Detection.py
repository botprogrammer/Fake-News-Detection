#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Necessary Imports

import numpy as np
import pandas as pd
import sklearn as sk
import itertools

#Fetching Data from CSV file
df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Fake News Detection\\news.csv') # Path to 'news.csv' file

df.shape
df.head()


# ## Getting Labels and splitting them

# In[2]:


from sklearn.model_selection import train_test_split

labels = df.label

#Splitting data into training and test set

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size = 0.2, random_state = 42)


# ## Initializing TfidfVectorizer

# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

#Feature Scaling data
fs_xtrain = vectorizer.fit_transform(x_train)
fs_xtest = vectorizer.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(fs_xtrain, y_train)


# # Making Predictoins:

# In[4]:


from sklearn.metrics import accuracy_score

y_pred = pac.predict(fs_xtest)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy = {round(score*100, 2)}%')


# # Building Confusion Matrix:

# In[5]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print(cm)