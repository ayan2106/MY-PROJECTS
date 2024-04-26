#!/usr/bin/env python
# coding: utf-8
# Heart Diseases Diagnostics Analysis
# In[1]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Display all the columns of dataframe
pd.pandas.set_option('display.max_columns', None)


# In[2]:


#load the dataset
df = pd.read_csv('heart_disease_dataset.csv')


# In[3]:


#print shape of dataset with rows and columns
df.shape


# In[4]:


#show the first 5 rows of dataset
df.head()


# In[5]:


# To show all the features in dataset
df.columns


# ## There are total 14 features in this dataset.

# ![features%20description.jpg](attachment:features%20description.jpg)

# ### Attributes information : -
# https://archive.ics.uci.edu/ml/datasets/heart+disease

# In[83]:


# to check the missing values
df.isnull().sum()>0


# ### Obseravtion: there is no missing value in any of the column

# In[7]:


def heart_diseases(value):
    '''function to convert numerical feature to categorical feature'''
    if value == 0:
        return 'absence'
    else:
        return 'presnce'


# In[8]:


# adding new column to dataset of converted data

df['heart_disease'] = df['num'].apply(heart_diseases)


# In[9]:


df


# In[10]:


# to count the number of people having heart diseases and NOT having heart diseases

hd = df.groupby('heart_disease')['num'].count()
hd


# In[11]:


# to plot the bar chart of above using matplotlib and seaborn

plt.figure(figsize=(5,5))
clrs = sns.color_palette('bright')
explode = [0, 0.05]
plt.pie(hd, labels=['absence','presence'], autopct='%0.0f%%', colors=clrs, explode=explode)
plt.title('Heart Diseases Population by percentage')
plt.show()


# ### ⇒ From tha above observation, people having heart diseases (46%) are lesser than the people who do not have heart diseases(56%)

# In[36]:


# plotting countplot of population age using matplotlib and seaborn

plt.figure(figsize=(20,10))
plt.title('Population Age')
sns.countplot(x='age', data=df, palette='bright')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# ### ⇒ we can observe the count of population according to the their age eg. young, middle-age and elder people

# In[13]:


# To find the minimum, maximum and average of the population age using statistical analysis

min_age = df['age'].min()
max_age = df['age'].max()
mean_age = df['age'].mean()
print(f"The minimum age is {min_age}")
print(f"The maximum age is {max_age}")
print(f"The average age is {np.round(mean_age, 2)}")


# In[14]:


# To divide the population age in different categories

young_age = df[(df['age']>=29) & (df['age']<40)]
middle_age = df[(df['age']>=40) & (df['age']<55)]
old_age = df[(df['age']>55)]
print(f"Number of young age people = {len(young_age)}")
print(f'Number of middle age people = {len(middle_age)}')
print(f'Number of old age people = {len(old_age)}')


# In[15]:


# Bar plot using matplotlib and seaborn for different categories of population age

cat = ['young_age','middle_age','old_age']
_count = [len(young_age), len(middle_age), len(old_age)]
plt.bar(cat, _count, color=['green', 'blue','red'])
plt.title('Age category')
plt.xlabel('Age Range')
plt.ylabel('count')
plt.show()


# ### ⇒ From the above plot, we observed that old age population is more than the middle age and young age population.  And there is least population of young age.

# In[16]:


# converting numerical data into categorical data 

def age_range(row):
    '''converting population age into range of age'''
    if row>=29 and row<40:
        return 'youngAge'
    elif row>=40 and row<55:
        return 'middleAge'
    else:
        return 'oldAge'


# In[17]:


# applying converted data into our dataset

df['ageRange'] = df['age'].apply(age_range)
df.head()


# In[18]:


# converting numerical data into categorical data 

def _sex(row):
    if row==1:
        return 'male'
    else:
        return 'female'


# In[19]:


# applying converted data into our dataset

df['gender'] = df['sex'].apply(_sex)
df.head()


# In[41]:


# Scatter plot creation Age category vs gender using matplotlib

plt.figure(figsize=(12,10))
x=df.ageRange
y=df.age
sns.set(style='whitegrid', palette='bright')
sns.swarmplot(x, y, hue='gender', data=df, dodge=True, order = ['youngAge', 'middleAge', 'oldAge'])
plt.show()


# ### ⇒ In the given dataset, number of male population is more than the female population in each age group.

# In[34]:


# count plot for heart diseases according to thier age category

plt.figure(figsize=(12,8))
hue_orders = ['youngAge', 'middleAge', 'oldAge']
plt.title("Heart diseases according to thier age category")
sns.countplot(x='heart_disease', hue='ageRange', data=df, hue_order=hue_orders, palette='bright')
plt.xlabel("Heart Disease")
plt.ylabel('Counts')
plt.show()


# ### ⇒ Old age people are most affected by Heart disease and young age people are rarely affected.

# In[38]:


# count plot for heart diseases based on sex

plt.figure(figsize=(12,8))
plt.title("Heart diseases based on sex", fontsize=18)
sns.countplot(x=df['heart_disease'], hue='gender', data=df, palette="bright")
plt.xlabel("Heart Disease")
plt.ylabel('Counts')
plt.show()


# ### ⇒ From above graph, it is observed that there are more number of males affected by heart diseases comparison to females.

# In[40]:


# count plot based on chest pain experienced

plt.figure(figsize=(12,8))
# hue_orders = ['youngAge', 'middleAge', 'oldAge']
plt.title("Chest pain experienced", fontsize=18)
sns.countplot(x=df['heart_disease'], hue='cp', data=df, palette="bright")
plt.legend(labels=['typical angina','atypical angina','non-anginal pain','asymptomatic'])
plt.xlabel("Heart Disease")
plt.ylabel('Counts')
plt.show()


# ### ⇒ it is observed that people having asymptomatic chest pain have higher chance of heart diseases.

# In[46]:


# Count plot for chest pain based on gender

plt.figure(figsize=(10,7))
plt.title("Chest pain based on sex", fontsize=18)
sns.countplot('gender', hue='cp', data=df, palette='bright')
plt.legend(labels=['typical angina','atypical angina','non-anginal pain','asymptomatic'])
plt.xlabel('Sex', fontsize=15)
plt.ylabel('Count')
plt.show()


# ### It is observed that higher number of males are suffering from asymptotic chest pain.

# In[48]:


# Count plot for chest pain vs age group using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title("Chest pain based on age group", fontsize=18)
sns.countplot(x=df['ageRange'], hue='cp', data=df, palette='bright', order=['youngAge', 'middleAge', 'oldAge'])
plt.legend(labels=['typical angina','atypical angina','non-anginal pain','asymptomatic'])
plt.xlabel('Age Groups')
plt.ylabel('Count')
plt.show()


# ### It seems that old age group have very high asymptomatic chest pain.

# In[51]:


# Bar graph for Restin blood pressure(trestbs)(in mm Hg) based on gender using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Blood pressure based on gender', fontsize=18)
sns.barplot(x='gender', y='trestbps', data=df, palette='bright')
plt.xlabel('Sex')
plt.ylabel('Blood pressure (in mm Hg)')
plt.show()


# ### It is observed that Person's Resting Blood Pressue is almost same for males and females.

# In[52]:


# Bar graph for Cholestral level based on gender using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Cholestral Level based on gender', fontsize=18)
sns.barplot(x='gender', y='chol', data=df, palette='bright')
plt.xlabel('Sex')
plt.ylabel('Cholestral level')
plt.show()


# ### Cholestral level is little bit more in females as compared to males.

# In[54]:


# Bar graph for Cholestral level vs Heart diseases using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Cholestral Level Vs Heart diseases', fontsize=18)
sns.barplot(x='heart_disease', y='chol', data=df, palette='bright')
plt.xlabel('heart_disease')
plt.ylabel('Cholestral level')
plt.show()


# ### Person having high cholestral level having high chance of heart diseases.

# In[55]:


# Bar graph for Blood pressure vs Heart diseases using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Blood pressure Vs Heart diseases', fontsize=18)
sns.barplot(x='heart_disease', y='trestbps', data=df, palette='bright')
plt.xlabel('heart_disease')
plt.ylabel('Blood pressure in mmHg')
plt.show()


# ### Higher blood pressure leads to high chances of heart diseases.

# In[58]:


# Line plot for blood pressue vs age using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Blood pressure Vs Age', fontsize=18)
sns.lineplot(x='age', y='trestbps',data=df, palette='bright')
plt.xlabel('Age')
plt.ylabel('Blood Pressure in mmHg')
plt.show()


# ### Here we observed that the blood pressure is normal from 30 to 50 and after 50 it increases gradually  to age of 60. And after age of 60 it is fluctuating drastically.

# In[61]:


# Line plot for cholestral level vs age using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Cholestral level Vs Age', fontsize=18)
sns.lineplot(x='age', y='chol',data=df, color='green')
plt.xlabel('Age')
plt.ylabel('cholestral level')
plt.show()


# ### Cholestral level is start increasing at the age of 50.

# In[62]:


# Line plot for ST depression vs age using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('ST depression Vs Age', fontsize=18)
sns.lineplot(x='age', y='oldpeak',data=df, color='red')
plt.xlabel('Age')
plt.ylabel('ST depression')
plt.show()


# ### Depression level is quite high in the age group of 30 to 40 and 55 to 70. And in the age group of 40 to 55, it remains stable.

# In[63]:


# Bar graph for ST Depression vs Heart diseases using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('ST Depression Vs Heart diseases', fontsize=18)
sns.barplot(x='heart_disease', y='oldpeak', data=df, palette='bright')
plt.xlabel('heart_disease')
plt.ylabel('ST Depression')
plt.show()


# ### People with high ST depression having higher chances of Heart diseases.

# In[64]:


# Bar graph for ST Depression vs Gender using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('ST Depression Vs Sex', fontsize=18)
sns.barplot(x='gender', y='oldpeak', data=df, palette='bright')
plt.xlabel('Sex')
plt.ylabel('ST Depression')
plt.show()


# ### It is observed that more number of males are prone to ST Depression as compare to females.

# In[68]:


# Bar graph for exercise induced Angina vs Heart diseases using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Exercise induced angina  Vs Heart diseases', fontsize=18)
sns.barplot(x='heart_disease', y='exang', data=df, palette='bright')
plt.xlabel('heart_disease')
plt.ylabel('Exercise induced Angina')
plt.show()


# ### It is observed that if the people alredy suffered with angina then exercise will make it worse condition for him/her.

# In[70]:


# Bar graph for Exercise induced Angina vs Gender using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Exercise induced Angina Vs Sex', fontsize=18)
sns.barplot(x='gender', y='exang', data=df, palette='bright')
plt.xlabel('Sex')
plt.ylabel('Exercise induced Angina')
plt.show()


# ###  Male having high exercise induced Angina.

# In[71]:


# Bar graph for Fasting blood sugar vs Gender using matplotlib and seaborn

plt.figure(figsize=(10,7))
plt.title('Fasting blood sugar Vs Sex', fontsize=18)
sns.barplot(x='gender', y='fbs', data=df, palette='bright')
plt.xlabel('Sex')
plt.ylabel('Fasting blood sugar')
plt.show()


# ### It is observed that males having high fasting blood pressure>120mg/dl

# In[82]:


# Heatmap usng seaborn

plt.figure(figsize=(16,8))
sns.heatmap(df.corr(), annot=True, linewidths=2, linecolor='white', cmap="Greens")


# In[84]:


df


# In[86]:


# exporting final dataset to as csv file for Dashboarding in Power BI

df.to_csv("D:\internship\my_data.csv", index=False)


# In[ ]:




