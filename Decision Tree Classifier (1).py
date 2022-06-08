#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import category_encoders as encod
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[2]:


path = "C:/Users/Admin/OneDrive/Desktop/D_Tree.xlsx"
data = pd.read_excel(path)
for i in data.columns:
    print(i)


# In[3]:


data.shape


# In[4]:


data.tail()


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data['Rain Today'].unique()


# In[8]:


data['Rain Today'].value_counts()


# In[9]:


sns.countplot(data=data,x='Rain Today')


# In[10]:


sns.boxplot(data=data)


# In[11]:


#Dropping the outliers from Tmax,Tmin,Totprep
data = data.drop(data[(data['maxtempC']<20)|(data['maxtempC']>38)].index)

data = data.drop(data[(data['mintempC']<15)|(data['mintempC']>27)].index)
data = data.drop(data[(data['totalprecipIn']>0.2)|(data['totalprecipIn']<-0.1)].index)



# In[12]:


sns.boxplot(data=data)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X=data.drop(['Rain Today','date','weatherDesc'],axis=1)
y=data['Rain Today']
y=y.astype('str')
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2)


# In[15]:


X


# In[16]:


y.unique()


# In[17]:


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# In[20]:


models=[]
models.append(('CART', DecisionTreeClassifier()))




# In[21]:


results=[]
names=[]
for name, model in models:
    kfold=KFold(n_splits=10, shuffle=True, random_state=3)
    cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg=(name,round((cv_results.mean())*100),cv_results.std())
    print(msg)
    


# In[22]:


#Prediction

model = DecisionTreeClassifier()
model.fit(X,y)

prediction = model.predict([[31,31,0.11]])

int(prediction[0])


# In[ ]:





# In[ ]:




