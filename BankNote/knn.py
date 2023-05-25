#!/usr/bin/env python
# coding: utf-8

# In[337]:


import numpy as np
import pandas as pd
import sys
from math import sqrt
from bisect import bisect


# In[ ]:





# In[338]:


data = pd.read_csv('C:/Users/citymall/Downloads/assigments/ML/BankNote_Authentication.csv')

#data.describe()
data


# In[ ]:





# In[339]:


#shuffling

#data=data.sample(frac=1)



x=data.iloc[:,[0,1,2,3]]
y=data.iloc[:,-1]

vr_std=np.std(x.iloc[:,0])
vr_mean=np.mean(x.iloc[:,0])

sk_std=np.std(x.iloc[:,1])
sk_mean=np.mean(x.iloc[:,1])

cort_std=np.std(x.iloc[:,2])
cort_mean=np.mean(x.iloc[:,2])

entr_std=np.std(x.iloc[:,3])
entr_mean=np.mean(x.iloc[:,3])

# normalizing data 
x.iloc[:,0]=(x.iloc[:,0]-vr_mean)/(vr_std)
x.iloc[:,1]=(x.iloc[:,1]-sk_mean)/(sk_std)
x.iloc[:,2]=(x.iloc[:,2]-cort_mean)/(cort_std)
x.iloc[:,3]=(x.iloc[:,3]-entr_mean)/(entr_std)
x


# In[340]:


data


# In[341]:


x


# In[342]:


res=data.iloc[:,[4]]
res


# In[343]:



x_train=x[0:961]
y_train=y[0:961]
x_test=x[961:]
y_test=y[961:]


# In[344]:



# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(0,len(row1)):
        distance += (row1[i] - row2[i])**2
        
    return sqrt(distance)    
 
 


# In[345]:


euclidean_distance(x_test.iloc[0,[0,1,2,3]], x_train.iloc[1,[0,1,2,3]])


# In[361]:


def check_neighbour(k,d,distances,y,lis):
    for x in range(0,k):
        if (x==k-1 and d< distances[x])or (d< distances[x]and distances[x]==5000000000):
            distances[x]=d
            lis[x] = y
            break
        elif d<distances[x]and distances[x]!=5000000000:
            for j in range(x,k-1):
               
                
                tmp=distances[j]
                distances[j]=d
                distances[j+1]=tmp
               

                tmp2 = lis[j]
                lis[j]=y
                lis[j+1]=tmp2
            break
                
            
           
    return distances,lis      
    
        


# In[347]:


def check_neighbour2(k,d,distances,y,lis):
  
           
    return distances,lis      


# In[348]:


def predict(lis):
    cnt_zero=0
    cnt_one=0
    for i in range(0, len(lis)):
        if lis[i]==0 :
            cnt_zero+=1
        else:
            cnt_one+=1
            
    #handle tie
    if cnt_zero==cnt_one:
        return lis[0]
        
    if cnt_zero>cnt_one:
        return 0
    else:
        return 1
        
                   


# In[349]:





# In[350]:





# In[362]:


def knn(k):
    lis=[]
    distances=[]
    for i in range(0, k):
        distances.append(5000000000)
        lis.append(-1)
        
    
    y_pred=[]
    print(distances,lis)

    for i in range(0,len(y_test)):
        for j in range(0,961):
            
            d= euclidean_distance(x_test.iloc[i,[0,1,2,3]], x_train.iloc[j,[0,1,2,3]])
            distances1,lis1= check_neighbour(k,d,distances,y_train[j],lis)
            distaces=distances1
            lis=lis1
           
            
        y_pred.append(predict(lis))
        
        
            
    return y_pred

        
            


# In[352]:


def accuracy(y_test,y_predict):
    sum=0
    for i in range(0,len(y_test)):
        if y_predict[i]==y_test[i]:
            sum+=1
        
    return sum, sum/len(y_test)


# In[363]:


y_pred = knn(5)

#


# In[ ]:





# In[ ]:





# In[ ]:


print()


# In[ ]:


print("k value", 3)
print("Number of correctly classified instances : ", sum(y_pred==y_test))
print("Total number of instances : " , len(y_test))
print("Accuracy : " , sum(y_pred==y_test)/len(y_test))


# In[ ]:




