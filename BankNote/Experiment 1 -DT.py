#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot
import csv


# In[2]:


dataset = pd.read_csv('D:/Final/Machine/assignment_2/BankNote_Authentication.csv') 
dataset


# In[3]:


from sklearn.model_selection import train_test_split
x=dataset.iloc[:,[0,1,2,3]]
y=dataset.iloc[:,[4]]


# In[13]:


# splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 50, train_size = 0.25) 

# create DecisionTreeClassifier obj and fitting data 
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

nodes_num = clf.tree_.node_count    # number of nodes node  (tree size)

#make a prediction and calculate accuracy (train - test)
y_predict =clf.predict(x_test)
print("Train data accuracy:",accuracy_score(y_train, y_pred=clf.predict(x_train)))
print("Test data accuracy:",accuracy_score( y_test, y_predict))
print("size of the tree :",nodes_num)


# In[12]:


# splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 100, train_size = 0.25) 

# create DecisionTreeClassifier obj and fitting data 
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

nodes_num = clf.tree_.node_count    # number of nodes node  (tree size)

#make a prediction and calculate accuracy (train - test)
y_predict =clf.predict(x_test)
print("Train data accuracy:",accuracy_score(y_train, y_pred=clf.predict(x_train)))
print("Test data accuracy:",accuracy_score( y_test, y_predict))
print("size of the tree :",nodes_num)


# In[11]:


# splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 150, train_size = 0.25) 

# create DecisionTreeClassifier obj and fitting data 
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

nodes_num = clf.tree_.node_count    # number of nodes node  (tree size)

#make a prediction and calculate accuracy (train - test)
y_predict =clf.predict(x_test)
print("Train data accuracy:",accuracy_score(y_train, y_pred=clf.predict(x_train)))
print("Test data accuracy:",accuracy_score( y_test, y_predict))
print("size of the tree :",nodes_num)


# In[10]:


# splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 200, train_size = 0.25) 

# create DecisionTreeClassifier obj and fitting data 
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

nodes_num = clf.tree_.node_count    # number of nodes node  (tree size)

#make a prediction and calculate accuracy (train - test)
y_predict =clf.predict(x_test)
print("Train data accuracy:",accuracy_score(y_train, y_pred=clf.predict(x_train)))
print("Test data accuracy:",accuracy_score( y_test, y_predict))
print("size of the tree :",nodes_num)


# In[9]:


# splitting data
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 250, train_size = 0.25) 

# create DecisionTreeClassifier obj and fitting data 
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

nodes_num = clf.tree_.node_count    # number of nodes node  (tree size)

#make a prediction and calculate accuracy (train - test)
y_predict =clf.predict(x_test)
print("Train data accuracy:",accuracy_score(y_train, y_pred=clf.predict(x_train)))
print("Test data accuracy:",accuracy_score( y_test, y_predict))
print("size of the tree :",nodes_num)


# In[ ]:




