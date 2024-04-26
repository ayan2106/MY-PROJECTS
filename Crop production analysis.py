#!/usr/bin/env python
# coding: utf-8

# <font size="6"> **Data Cleaning** </font>

# <font size="4"> Importing essential Libraries </font>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from mpl_toolkits import mplot3d


# <font size="4"> Creating a Dataframe using Pandas to read the raw data file. </font>

# In[3]:


df = pd.read_csv("crop_production.csv")


# <font size="4">Names of all columns in the Dataframe</font>

# In[4]:


df.columns


# In[5]:


df.head()


# <font size="4">Checking the datatypes of all the columns in the dataframe</font>

# In[6]:


df.info()


# <font size="4"> Creating a new column named "Date" having the datatype of date </font>

# In[7]:


df["Date"] = pd.to_datetime(df["Crop_Year"], format='%Y')


# In[8]:


df.head()


# <font size="4">All unique state names</font>

# In[9]:


Unique_State_Name = df["State_Name"].unique()
print(Unique_State_Name)


# In[10]:


Unique_Date = df["Date"].unique()
print(Unique_Date)


# In[11]:


df["Season"] = df["Season"].str.strip()
Unique_Season = df["Season"].unique()
print(Unique_Season)


# <font size="4">Categorizing the 6 seasons into 4 major Indian cropping seasons</font>

# In[12]:


df["Season"] = df["Season"].apply( lambda x : "Rabi" if x == "Winter" else  x )
df["Season"] = df["Season"].apply( lambda x : "Rabi" if x == "Autumn" else  x )
df["Season"] = df["Season"].apply( lambda x : "Zaid" if x == "Summer" else  x )
print(df["Season"].unique())


# <font size="4">All unique crop names</font>

# In[13]:


df["Crop"] = df["Crop"].str.strip()
Unique_Crop = df["Crop"].unique()
print(Unique_Crop)


# <font size="4">Since many of the crop names were repeated,duplicated, or having wrong spelling.
# Combining all the duplicated or duplicated into one crop and correcting the spelling of all the incorrect ones.</font>

# In[14]:


df["Crop"] = df["Crop"].apply( lambda x : "Seasum" if x == "Sesamum" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Pomegranate" if x == "Pome Granet" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Sunhemp" if x == "Sannhamp" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Raddish" if x == "Redish" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Soybean" if x == "Soyabean" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Snake Gourd" if x == "Snak Guard" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Pumpkin" if x == "Pump Kin" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Paddy" if x == "Rice" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Blackgram" if x == "Urad" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Lentil" if x == "Masoor" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Jute & mesta" if x == "Jute" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Jute & mesta" if x == "Mesta" else  x )
df["Crop"] = df["Crop"].apply( lambda x : "Cotton(lint)" if x == "Kapas" else  x )

Unique_Crop = df["Crop"].unique()
print(Unique_Crop)


# <font size="4">India has a huge variety of crops grown.
# It would be easier to analyze if all the crops are further categorized into crop categories.</font>

# In[15]:


def category_name(name):
    Cereals = ['Wheat','Maize','Bajra','Paddy', 'Jowar', 'Korra','Ragi','Small millets','Samai', 'Varagu','Barley','Khesari','Other Cereals & Millets','Total foodgrain','Jobster']
    Pulses = ['Pulses total','Other  Rabi pulses','Other Kharif pulses','Moong(Green Gram)','Blackgram','Lentil','Arhar/Tur','Horse-gram','Gram','other misc. pulses','Other rabi pulses','Other fibres','Cowpea(Lobia)','Peas & beans (Pulses)','Moth','Bean','Rajmash Kholar','Ricebean (nagadal)']
    Vegetables = ['Beans & Mutter(Vegetable)','Lemon','Sweet potato','Tapioca','Onion','Potato','Bhindi','Brinjal','Cucumber','Other Vegetables','Tomato','Cabbage','Peas  (vegetable)','Bottle Gourd','Turnip','Carrot','Raddish','Bitter Gourd','Drum Stick','Jack Fruit','Snake Gourd','Pumpkin','Cauliflower','Colocosia','Ash Gourd','Beet Root','Lab-Lab','Ribed Guard','Yam','Perilla']
    Fruits = ['Sapota','Banana','Citrus Fruit','Grapes','Mango','Orange','Other Fresh Fruits','Papaya','Pome Fruit','Pomegranate','Pineapple','Other Citrus Fruit','Water Melon','Apple','Peach','Pear','Plums','Litchi','Ber']
    Oilseeds = ['other oilseeds','Groundnut','Sunflower','Castor seed','Safflower','Linseed','Seasum','Rapeseed &Mustard','Niger seed','Oilseeds total','Sunhemp','Soybean']
    Condiments = ['Dry ginger','Black pepper','Dry chillies','Turmeric','Coriander','Garlic','Ginger','Cond-spcs other','Cardamom','Arcanut (Processed)','Atcanut (Raw)','Arecanut']
    Dryfruits = ['Cashewnut','Cashewnut Processed','Cashewnut Raw','Other Dry Fruit']
    Plantation = ['Coconut','Sugarcane','Cotton(lint)','Tobacco','Jute & mesta','Guar seed','Rubber','Tea','Coffee','other fibres']
    if name in Cereals:
        return 'Cereal'
    elif name in Pulses:
        return 'Pulses'
    elif name in Vegetables:
        return 'Vegetable'
    elif name in Fruits:
        return 'Fruit'
    elif name in Oilseeds:
        return 'Oilseed'
    elif name in Condiments:
        return 'Condiment'
    elif name in Dryfruits:
        return 'Dryfruit'
    elif name in Plantation:
        return 'Plantation'
    else:
        return 'None'
    
df["Category"] = df["Crop"].apply(category_name)
print(df["Category"].unique())


# <font size="4">Categorizing all the indian states into their respected regions.</font>

# In[16]:


def category(name):
    east_india = ['West Bengal', 'Tripura', 'Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Sikkim','Odisha']
    west_india = ['Maharashtra', 'Gujarat', 'Dadra and Nagar Haveli', 'Goa', 'Rajasthan']
    north_india = ['Bihar', 'Chandigarh', 'Chhattisgarh', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir ', 'Jharkhand', 'Madhya Pradesh', 'Punjab', 'Uttar Pradesh', 'Uttarakhand']
    south_india = ['Tamil Nadu', 'Telangana ', 'Andaman and Nicobar Islands', 'Andhra Pradesh', 'Karnataka', 'Kerala', 'Puducherry']
    
    if name in east_india:
        return 'East India'
    elif name in west_india:
        return 'West India'
    elif name in north_india:
        return 'North India'
    elif name in south_india:
        return 'South India'
    else:
        return 'None'
    
df["Region"] = df["State_Name"].apply(category)
print(df["Region"].unique())


# In[17]:


ny_df = df.loc[df['Region'] == 'None']
ny_df["State_Name"].unique()


# In[18]:


df.columns = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop', 'Area(hectare)',
       'Production(tonnes)', 'Date', 'Category', 'Region']


# In[19]:


df.columns


# <font size="4">Creating a new column named "roductivity" and "Total Production"</font>

# In[20]:


df["Productivity"] = df["Production(tonnes)"] / df["Area(hectare)"]
df["Total Production"] = df["Production(tonnes)"] * df["Area(hectare)"]


# In[21]:


df.columns


# In[22]:


df.head()


# <font size="4">Dropping all the null values</font>

# In[23]:


df = df.dropna(axis=0)


# <font size="4">Dropping all the rows in which Area is 1,2,3,4,5,6,7 and 10</font>

# In[24]:


df = df.drop(df[df['Area(hectare)'] == 1].index)
df = df.drop(df[df['Area(hectare)'] == 2].index)
df = df.drop(df[df['Area(hectare)'] == 3].index)
df = df.drop(df[df['Area(hectare)'] == 4].index)
df = df.drop(df[df['Area(hectare)'] == 5].index)
df = df.drop(df[df['Area(hectare)'] == 6].index)
df = df.drop(df[df['Area(hectare)'] == 7].index)
df = df.drop(df[df['Area(hectare)'] == 10].index)


# <font size="4">Dropping all the rows in which Production is 0,1,2,3,4,5,6 and 10</font>

# In[25]:


df = df.drop(df[df['Production(tonnes)'] == 1].index)
df = df.drop(df[df['Production(tonnes)'] == 0].index)
df = df.drop(df[df['Production(tonnes)'] == 2].index)
df = df.drop(df[df['Production(tonnes)'] == 3].index)
df = df.drop(df[df['Production(tonnes)'] == 4].index)
df = df.drop(df[df['Production(tonnes)'] == 5].index)
df = df.drop(df[df['Production(tonnes)'] == 6].index)
df = df.drop(df[df['Production(tonnes)'] == 10].index)


# In[26]:


df.to_csv("Cleaned.csv", index = False)


# In[27]:


df.columns


# <font size="4">Removing the outliers</font>

# In[28]:


Q1 = df["Total Production"].quantile(0.40)
Q3 = df["Total Production"].quantile(0.60)
IQR = Q3 - Q1

df = df[(df["Total Production"] >= Q1 - 1.5*IQR) & (df["Total Production"] <= Q3 + 1.5*IQR)]


# In[29]:


Q1 = df["Productivity"].quantile(0.40)
Q3 = df["Productivity"].quantile(0.60)
IQR = Q3 - Q1

df = df[(df["Productivity"] >= Q1 - 1.5*IQR) & (df["Productivity"] <= Q3 + 1.5*IQR)]


# In[30]:


plt.hist(df["Total Production"])


# In[31]:


plt.hist(df["Productivity"])


# In[32]:


df.to_csv("CleanedIQR.csv", index = False)


# In[33]:


Q1 = df["Area(hectare)"].quantile(0.40)
Q3 = df["Area(hectare)"].quantile(0.60)
IQR = Q3 - Q1

df = df[(df["Area(hectare)"] >= Q1 - 1.5*IQR) & (df["Area(hectare)"] <= Q3 + 1.5*IQR)]


# In[34]:


plt.hist(df["Area(hectare)"])


# In[35]:


df.to_csv("CleanedIQR.csv", index = False)

