#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df= pd.read_excel(r'C:\Users\ayann\Downloads\Bird Strikes data.xlsx')


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.ndim


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df['Remarks'].unique()


# In[10]:


df['Origin State'].unique()


# In[12]:


df['When: Phase of flight'].unique()


# In[13]:


df['Aircraft: Type'].unique()


# In[14]:


df.columns


# In[15]:


df.size


# In[17]:


df.info()


# In[19]:


df1 = df.copy()


# In[20]:


df1


# In[21]:


df1.dropna(subset=['Aircraft: Type','Airport: Name', 'Altitude bin','Wildlife: Number struck','Effect: Impact to flight','FlightDate','Aircraft: Number of engines?','Aircraft: Airline/Operator','Origin State','When: Phase of flight','Wildlife: Size','Pilot warned of birds or wildlife?','Feet above ground','Is Aircraft Large?'],inplace=True)


# In[23]:


df1.isnull().sum()


# In[24]:


df1.describe()


# In[25]:


df1.drop(labels='Remarks', axis=1,inplace=True)


# In[26]:


df1.isnull().sum()


# In[27]:


df1


# In[28]:


df1.to_excel(r'C:\Users\ayann\Downloads\Bird Strikes data.xlsx',index=False)


# In[ ]:




