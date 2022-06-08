#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
import os


# In[2]:


path = "C:/Users/Admin/OneDrive/Desktop/Project Info/wd_24hr - Copy.csv"
data = pd.read_csv(path)
for i in data.columns:
    print(i)


# In[3]:


data.info


# In[4]:


# Missing Data Pattern in Data
import seaborn as sns
sns.heatmap(data.isnull(), cbar=False, cmap='PuBu')


# In[5]:


mno.bar(data)


# Hence No missing data

# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


def func(x):
    if x=='Cloudy':
        return 1
    elif x=='Partly cloudy':
        return 2
    elif x=='Sunny':
        return 3
    elif x=='Patchy light drizzle':
        return 4
    elif x=='Light rain shower':
        return 5
    elif x=='Moderate or heavy rain shower':
        return 6
    elif x=='Patchy rain possible':
        return 7
    elif x=='Heavy rain':
        return 8
    elif x=='Patchy light rain':
        return 9
    elif x=='Heavy rain at times':
        return 10
    elif x=='Light drizzle':
        return 11
    elif x=='Light rain':
        return 12
    elif x=='Mist':
        return 13
    elif x=='Moderate rain':
        return 14
    elif x=='Moderate rain at times':
        return 15
    elif x=='Overcast':
        return 16
    elif x=='Patchy light rain with thunder':
        return 17
    elif x=='Thundery outbreaks possible':
        return 18
    elif x=='Torrential rain shower':
        return 19
    

data['Ordinals'] = data['weatherDesc'].apply(func)


# In[9]:


data.head()


# In[10]:


data["Ordinals"].unique()


# In[11]:


data.columns


# In[12]:


type(data['Ordinals'])


# In[13]:


data.iloc[:,5]


# In[14]:


#Breaking down Independent and Dependent variables
X = data.iloc[: , 1:5].values #upperbound is omitted
y = data.iloc[:, 4].values.astype(int)


# In[15]:


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[16]:


from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler(feature_range=(0,1))
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)


# In[17]:


import warnings
warnings.filterwarnings("ignore")
#Fitting the Classifier
from sklearn.svm import SVC
classifier = SVC(kernel='poly', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting rain/no
y_pred = classifier.predict(X_test)

#Analysing our model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print("Accuracy:",round((scores.mean())*100),"%") 


# In[ ]:





# In[ ]:




