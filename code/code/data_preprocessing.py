#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pyspark
from pyspark.sql import SparkSession


# In[27]:


import pandas as pd
import numpy as np
import time


# In[28]:


#read from csv and store as dataframe
filename = "XRP"
df = pd.read_csv("./raw_data/coin_"+filename+".csv")
df


# In[29]:


#calculate for relevant weekly records(max,min or avg)
#return a tuple contains all the calculation results
def cal_week(l):
    n = len(l)
    h = l[0][0]
    low = l[0][1]
    
    s = 0
    s1 = 0
    s2 = 0
    s3 = 0
    
    for i in range(0,n):
        if h < l[i][0]:   #maximum for highest price
            h = l[i][0]
        if low > l[i][1]:   #minimum for lowest price
            low = l[i][1]
        
        #average
        s = s+l[i][4]   #average volumes
        s1 = s1+l[i][5] #average market cap
        s2 = s2+l[i][2] #average opening price
        s3 = s3+l[i][3] #average closing price
        
    t = (h,low,l[0][2],l[n-1][3],s/n,s1/n,s2/n,s3/n)
    return t
        


# In[30]:


#convert date into year,month and day respectively
n = len(df)
df = df.drop(['SNo'],axis=1)

#rename the date column as Year
df.rename(columns={'Date':'Year'},inplace=True)
#insert month and date column
df.insert(3,"Mon",np.zeros(n))
df.insert(4,"Day",np.zeros(n))
#insert average opening and closing price column
df.insert(11,"o_avg",np.zeros(n))
df.insert(12,"c_avg",np.zeros(n))

for i in range(0,n):
    t = time.strptime(df.iat[i,2], "%Y-%m-%d %H:%M:%S")
    df.iat[i,2]=t.tm_year
    df.iat[i,3]=t.tm_mon
    df.iat[i,4]=t.tm_mday
#df  


# In[31]:


#split daily data and merge daily data into weekly data
newdf = pd.DataFrame(columns=df.columns)
n = len(df)

i = 0
k = 0 #index of the new dataframe

while(i<n):
    j = i
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    t = []
    t1 = []
    t2 = []
    t3 = []
    #print(1)
    while(j<n):
        #print(2,end = " ")
        if df.iat[j,2] != df.iat[i,2] or df.iat[j,3] != df.iat[i,3]:
            break
        else: #the same month and the same year
            t_tuple = (df.iat[j,5],df.iat[j,6],df.iat[j,7],df.iat[j,8],df.iat[j,9],df.iat[j,10])
            if df.iat[j,4]>=1 and df.iat[j,4]<=7: #1-7
                count = count+1
                t.append(t_tuple) 
            if df.iat[j,4]>=8 and df.iat[j,4]<=14: #8-14
                count1 = count1+1
                t1.append(t_tuple) 
            if df.iat[j,4]>=15 and df.iat[j,4]<=21: #15-21
                count2 = count2+1
                t2.append(t_tuple) 
            if df.iat[j,4]>=22 and df.iat[j,4]<=31: #22-31
                count3 = count3+1
                t3.append(t_tuple)
            j = j+1
    
    #update the values in the new dataframe
    if count != 0:
        newdf = newdf.append(df.iloc[i])
        a_t = cal_week(t)
        #print(a_t)
        newdf.iat[k,4] = 1
        (newdf.iat[k,5],newdf.iat[k,6],newdf.iat[k,7],newdf.iat[k,8],newdf.iat[k,9],newdf.iat[k,10],newdf.iat[k,11],newdf.iat[k,12])=a_t
        k = k+1
    if count1 != 0:
        newdf = newdf.append(df.iloc[i])
        a_t = cal_week(t1)
        #print(a_t)
        newdf.iat[k,4] = 2
        (newdf.iat[k,5],newdf.iat[k,6],newdf.iat[k,7],newdf.iat[k,8],newdf.iat[k,9],newdf.iat[k,10],newdf.iat[k,11],newdf.iat[k,12])=a_t
        k = k+1
    if count2 != 0:
        newdf = newdf.append(df.iloc[i])
        a_t = cal_week(t2)
        #print(a_t)
        newdf.iat[k,4] = 3
        (newdf.iat[k,5],newdf.iat[k,6],newdf.iat[k,7],newdf.iat[k,8],newdf.iat[k,9],newdf.iat[k,10],newdf.iat[k,11],newdf.iat[k,12])=a_t
        k = k+1
    if count3 != 0:
        newdf = newdf.append(df.iloc[i])
        a_t = cal_week(t3)
        #print(a_t)
        newdf.iat[k,4] = 4
        (newdf.iat[k,5],newdf.iat[k,6],newdf.iat[k,7],newdf.iat[k,8],newdf.iat[k,9],newdf.iat[k,10],newdf.iat[k,11],newdf.iat[k,12])=a_t
        k = k+1
        
    i = j


newdf.reset_index(drop=True,inplace=True) #reset the index of the new dataframe
                


# In[32]:


#process data for random forest model
#insert true value columns for predicton
n = len(newdf)
newdf.insert(13,"Price1",np.zeros(n))
newdf.insert(14,"Volume1",np.zeros(n))
newdf.insert(15,"Popularity",np.zeros(n))


# In[33]:


#add true values for predicton
for i in range(0,n):
    if i != n-1:
        newdf.iat[i,13] = newdf.iat[i+1,12] #the price in next row
        newdf.iat[i,14] = newdf.iat[i+1,9] #the volume in next row
        newdf.iat[i,15] = newdf.iat[i,13]*newdf.iat[i,14] #popularity = price*volume
        
#export to corresponding csv files, one cryptocurrency one file
newdf.to_csv("./processed_data/"+filename+".csv")
newdf


# In[ ]:




