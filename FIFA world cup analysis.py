#!/usr/bin/env python
# coding: utf-8

# #  FIFA WORLD CUP DATA ANALYSIS  (  Internship Project )

# ## Problem Statement:

# 
# * With FIFA being in the blood as many people of the world. You are tasked to tell the story of unsung analysts who put great efforts to provide accurate data to answer every question of fans.

# * The FIFA World Cup is a global football competition contested by the various football- playing nations of the world. It is contested every four years and is the most prestigious and important trophy in the sport of football.

# * The World Cups dataset show all information about all the World Cups in the history, while the World Cup Matches dataset shows all the results from the matches contested as part of the cups.

# * Find key metrics and factors that influence the World Cup win.Do your own research and come up with your findings.

# ## Solution Statement : 

# * In this project i am going to conduct an analysis on "FIFA-World-Cup" dataset witch can be accessed from iNeuron Internship     Project.

# *  Project Title       -  FIFA WORLD CUP 
# *  Technology          -  Business Intelligence
# *  Domain              -  Sports
# *  Difficulties Level  -  Advanced
# *  Dataset link        -  https://drive.google.com/drive/folders/12oHYj0qH2uZD8I13cVDiymTNDYldeJRa?usp=sharing 

# * The FIFA World Cup is a global football competition contested by the various football-playing nations of the world. 
# * It is contested every four years and is the most prestigious and important trophy in the sport of football. 
# * The World Cups dataset show all information about all the World Cups in the history, while the World Cup Matches dataset       shows all the results from the matches contested as part of the cups and World Cup Players dataset show all the information
#   about players coach.
# * This data is courtesy of the FIFA World Cup Archive website.

# * I will practice some analytical technic skills that i gathered from last 4 months.

# # Data Preparation and Cleaning

# * In this section I will be using some data preparation and data cleaning techniques to make my datasets easy and suitable to     use and having information about them (like knowing how many number of columns and rows they have, ranges of values etc). I     will start by loading them into a data frame with the help of Pandas library, and then perform further steps on them.
# * I prefer to start by importing all libraries that i think i will need

# # Imported the required Libraries

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
py.offline.init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# # Reading the WorldCup Matches data sets :-

# In[4]:


df_Matches = pd.read_csv("./Desktop/My-FIFA-Projects/WorldCupMatches.csv")


# In[5]:


# Displaying the top 5 rows of the dataframe(Matches)
display(df_Matches.head())

# Displaying the statistical information for the dataframe
display(df_Matches.describe())

# Displaying the information of the dataframe
display(df_Matches.info())

# Displaying the dimention of the dataframe
display ("Shape of Matches " , df_Matches.shape)
print(f"The Matches data set has {df_Matches.shape[0]} rows and {df_Matches.shape[1]} columns")


# In[6]:


# Checking if i have some null values
df_Matches.isnull().sum()


# * This shows that all columns have the same number of null values, whitch means that maybe they start at the same index and end at the same index. but i have to delete all of them

# In[7]:


# Visualization of null values

sns.heatmap(df_Matches.isnull(), cbar= True)


# * This shows that null values start at the same index, then i can know that index by checking the last valid index of the dataset
# 
# 

# In[8]:


df_Matches.last_valid_index()


# * This means that from the row 852, all values are NaN let me check that

# In[9]:


df_Matches[850:860]


# In[10]:


# Dropping all the null values

df_Matches = df_Matches.dropna()


# In[11]:


# Now rechecking for the null values

df_Matches.isnull().sum() 


# * Now we can see there are zero null values in the matchces dataset.

# => Now i am going to check if there is some data redundance, then delete them if exist to make sure that no duplicates in my      Matches datasets.

# In[12]:


df_Matches_dup = df_Matches[df_Matches.duplicated(keep = "last")]
df_Matches_dup


# In[13]:


df_Matches_dup1 = df_Matches[df_Matches.duplicated(keep = "first" )]
df_Matches_dup1


# In[14]:


# Shape of duplicate values

df_Matches_dup.shape


# * This shows that i have 15 rows duplicates data, so i have to remove them.

# In[15]:


# Removing duplicate values 

df_Matches = df_Matches.drop_duplicates()


# In[16]:


# seeing the dimentions of data frames(MATCHES) afteer dropping null and duplicate values

display ("Shape of df_Matches " , df_Matches.shape)
print(f"The Matches data set has {df_Matches.shape[0]} rows and {df_Matches.shape[1]} columns")


# * Now World Cup Matches dataset does not contain any null and duplicate values.
# * So i m going to remove columns that i think i will not need.

# In[17]:


# Showing all the available columns in the Matches dataset

df_Matches.columns


# In[18]:


del df_Matches["RoundID"]


# ### Now I am going to do small analysis on Players and Cups dataset.

# # Reading the WorldCup players data set :-

# In[19]:


df_Players = pd.read_csv('./Desktop/My-FIFA-Projects/WorldCupPlayers.csv')


# In[20]:


# Displaying the top 5 rows of the dataframe( Players )
display(df_Players.head())

# Displaying the statistical information for the dataframe
display(df_Players.describe())

# Displaying the information of the dataframe
display(df_Players.info())

# Displaying the dimentions of data frames(PLAYERS)

display ("Shape of df_Players " , df_Players.shape)
print(f"The Players data set has {df_Players.shape[0]} rows and {df_Players.shape[1]} columns")


# In[21]:


# Checking if i have some null values
df_Players.isnull().sum()


# In[22]:


# Visualization of null values
sns.heatmap(df_Players.isnull(), cbar= True)


# * This graph is showing that we have null values in position and event columns only.

# In[23]:


# Dropping all the null values
df_Players = df_Players.dropna()


# => Now i am going to check if there is some data redundance, then delete them if exist to make sure that no duplicates in my      Players datasets.

# In[24]:


df_Players_dup = df_Players[df_Players.duplicated(keep = "last")]
df_Players_dup


# In[25]:


df_Players_dup1 = df_Players[df_Players.duplicated(keep = "first")]
df_Players_dup1


# In[26]:


# Shape of duplicate values in Players dataset

df_Players_dup.shape


# * This shows that i have 18 rows duplicates data, so i have to remove them.

# In[27]:


# Removing duplicate values 

df_Players = df_Players.drop_duplicates()


# In[28]:


# Displaying the dimentions of data frames(PLAYERS) after dropping null and duplicate values

display ("Shape of df_Players " , df_Players.shape)
print(f"The Players data set has {df_Players.shape[0]} rows and {df_Players.shape[1]} columns")


# # Reading the WorldCup Cups data set :-

# In[29]:


df_Cups = pd.read_csv('./Desktop/My-FIFA-Projects/WorldCups.csv')


# In[30]:


# Displaying the top 5 rows of the dataframe ( Cups )
display(df_Cups.head())

# Displaying the statistical information for the dataframe
display(df_Cups.describe())

# Displaying the information of the dataframe
display(df_Cups.info())


# Displaying the dimentions of data frames(Cups)

display ("Shape of df_Cups " , df_Cups.shape)
print(f"The Cups data set has {df_Cups.shape[0]} rows and {df_Cups.shape[1]} columns")


# In[31]:


# Checking for the null values in Cups dataset
df_Cups.isnull().sum()


# * This shows us that Cups dataset does't have any null values

# In[32]:


class color:
   BOLD = '\033[1m'
   END = '\033[0m'
def show_dtypes(x):
    for col in x.columns:
        print(color.BOLD +col+ color.END,"::",x[col].dtypes)


# In[33]:


# Displaying data types in {Matches} data set

display("Dtypes of Matches")
print("------------------------------")
print("==============================")
display(show_dtypes(df_Matches))


# In[34]:


# Displaying data types in {Players} data set

display("Dtypes of Players")
print("------------------------------")
print("==============================")
display(show_dtypes(df_Players))


# In[35]:


# Displaying data types in {Cups} data set

display("Dtypes of Cups")
print("------------------------------")
print("==============================")
display(show_dtypes(df_Cups))


# In[36]:


from scipy import stats
def Tabulation(x):
    table = pd.DataFrame(x.dtypes,columns=['dtypes'])
    table1 =pd.DataFrame(x.columns,columns=['Names'])
    table = table.reset_index()
    table= table.rename(columns={'index':'Name'})
    table['No of Missing'] = x.isnull().sum().values    
    table['No of Uniques'] = x.nunique().values
    table['Percent of Missing'] = ((x.isnull().sum().values)/ (x.shape[0])) *100
    table['First Observation'] = x.loc[0].values
    table['Second Observation'] = x.loc[1].values
    table['Third Observation'] = x.loc[2].values
    for name in table['Name'].value_counts().index:
        table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2),2)
    return table


# In[37]:


Tabulation(df_Matches)


# In[38]:


Tabulation(df_Players)


# In[ ]:


Tabulation(df_Cups)


# In[39]:


def show_unique(x):
    class color:
        BOLD = '\033[1m'
        END = '\033[0m'
    for col in x.columns:
        print("Total Unique  " + color.BOLD +col+ color.END + "  values in Train Data {}".format(x[col].nunique()))


# In[40]:


show_unique(df_Matches)


# In[41]:


show_unique(df_Players)


# In[42]:


show_unique(df_Cups)


# In[43]:


def Attendence_by_year_bar_plot(x):
    x.isnull().sum()
    sns.set_style("darkgrid")

    # removing duplicate rows for match id column
    matches = x.drop_duplicates(subset="MatchID",keep="first")

    #removing null values
    matches = matches[matches["Year"].notnull()]

    att = matches.groupby("Year")["Attendance"].sum().reset_index()
    att["Year"] = att["Year"].astype(int)
    plt.figure(figsize=(12,7))
    sns.barplot(att["Year"],att["Attendance"],linewidth=1,edgecolor="k"*len(att))
    plt.xlabel("Year", fontsize = 20)
    plt.ylabel("Attendence", fontsize = 20)
    plt.grid(True)
    plt.title("Attendance In Every World Cup Year" , fontsize = 25 , color = "green")
    plt.show()
Attendence_by_year_bar_plot(df_Matches)


# In[44]:


def Average_Attendance_by_year(x):
    att1 = x.groupby("Year")["Attendance"].sum().reset_index()
    att1["Year"] = att1["Year"].astype(int)
    plt.figure(figsize=(12,7))
    ax = sns.pointplot(att1["Year"],att1["Attendance"],color="w")
    ax.set_facecolor("k")
    plt.xlabel("Year", fontsize = 20)
    plt.ylabel("Attendence", fontsize = 20)
    plt.grid(True,color="grey",alpha=.3)
    plt.title("Average Attendance In Every World Cup year" , fontsize = 25 , color = "green")
    plt.show()
Average_Attendance_by_year(df_Matches)


# In[45]:


# Here i m showing total goals scored by year using scatter plot

def Total_Goals_Scored_by_year(x):
    plt.figure(figsize=(13,7))
    x["Year1"] = x["Year"].astype(str)
    
    ax = plt.scatter("Year1","GoalsScored",
                     data=df_Cups,
                     c=df_Cups["GoalsScored"],
                     cmap="inferno",
                     s=900,
                     alpha=.7,
                     linewidth=2,
                     edgecolor="k")

    plt.xticks(x["Year1"].unique())
    plt.yticks(np.arange(60,200,20))
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Total Goal Scored",fontsize=20)
    plt.title('Total goals scored by year', fontsize = 25 , color = "green")
    plt.show()
Total_Goals_Scored_by_year(df_Cups)


# In[46]:


def Qualifies_Teams(x):
    plt.figure(figsize=(12,7))
    
    sns.barplot(x["Year"],x["MatchesPlayed"],
                linewidth=1,
                edgecolor="k"*len(x),
                color="yellow",
                label="Total matches played")
    
    sns.barplot(x["Year"],x["QualifiedTeams"],
                linewidth=1,
                edgecolor="k"*len(x),
                color="cyan",
                label="Total qualified teams")
    
    plt.legend(loc="best",prop={"size":13})
    plt.title("Qualified teams by year")
    plt.grid(True)
    plt.show()
    
Qualifies_Teams(df_Cups)


# In[47]:


def Highest_Attendence(x):
    h_att = x.sort_values(by="Attendance",ascending=False)[:10]
    h_att = h_att[['Year', 'Datetime','Stadium', 'City', 'Home Team Name',
                  'Home Team Goals', 'Away Team Goals', 'Away Team Name', 'Attendance', 'MatchID']]
    h_att["Stadium"] = h_att["Stadium"].replace('Maracan� - Est�dio Jornalista M�rio Filho',"Maracanã Stadium")
    h_att["Datetime"] = h_att["Datetime"].str.split("-").str[0]
    h_att["mt"] = h_att["Home Team Name"] + " .Vs.  " + h_att["Away Team Name"]

    plt.figure(figsize=(8,7))
    ax = sns.barplot(y = h_att["mt"],
                     x = h_att["Attendance"],
                     palette="gist_ncar",
                     linewidth = .8,
                     edgecolor="k"*len(h_att))
    
    plt.xlabel("Attendance", fontsize=20)
    plt.ylabel("Teams Match", fontsize=20)
    plt.title("Highest Attendance in a Match of World Cup History", fontsize=24, color='orange')
    
    ticks = np.arange(0, 200000, 25000)
    labels = ["{}K".format(i//1000) for i in ticks]
    plt.xticks(ticks, labels) 
   
    plt.grid(True)
    for i,j in enumerate(" stadium : "+h_att["Stadium"]+" , Date :" + h_att["Datetime"]):
        ax.text(.7,i,j,fontsize = 12,color="white",weight = "bold")
    plt.show()
Highest_Attendence(df_Matches)


# In[48]:


df_Matches["Year"] = df_Matches["Year"]
df_Matches["Datetime"] =df_Matches["Datetime"].str.split("-").str[0]


df_Matches["Stadium"] = df_Matches["Stadium"].str.replace('Estadio do Maracana',"Maracanã Stadium")
df_Matches["Stadium"] = df_Matches["Stadium"].str.replace('Maracan� - Est�dio Jornalista M�rio Filho',"Maracanã Stadium")
std  = df_Matches.groupby(["Stadium","City"])["Attendance"].mean().reset_index().sort_values(by = "Attendance",ascending =False)


plt.figure(figsize=(12,8))
ax = sns.barplot(y = std["Stadium"][:14],
                 x = std["Attendance"][:14],
                 palette="cool",
                 linewidth=1,edgecolor = "k"*14)
plt.xlabel("Attendance", fontsize=20)
plt.ylabel("Stadium", fontsize=20)
plt.grid(True)

ticks = np.arange(0, 100000, 20000)
labels = ["{}K".format(i//1000) for i in ticks]
plt.xticks(ticks, labels) 
    
for i,j in enumerate("  City : " + std["City"][:14]):
    ax.text(.7,i,j,fontsize=14)

plt.title("Stadiums with highest average attendance" ,  fontsize=25, color='orange')
plt.show()


# In[49]:


mat_c = df_Matches["City"].value_counts().reset_index()
plt.figure(figsize=(10,8))
ax = sns.barplot(y = mat_c["index"][:15],
                 x = mat_c["City"][:15],
                 palette="gist_earth",
                 linewidth=1,edgecolor="k"*15)
plt.xlabel("number of matches")
plt.ylabel("City")
plt.grid(True)
plt.xlabel("Number Of Matches", fontsize=20)
plt.ylabel("City", fontsize=20)
plt.title("Cities with maximum world cup matches")

for i,j in enumerate("Matches  :" + mat_c["City"][:15].astype(str)):
    ax.text(.7,i,j,fontsize = 13,color="w")
plt.show()


# In[50]:


ct_at = df_Matches.groupby("City")["Attendance"].mean().reset_index()
ct_at = ct_at.sort_values(by="Attendance",ascending=False)
ct_at

plt.figure(figsize=(8,8))

ax = sns.barplot("Attendance","City",
            data=ct_at[:20],
            linewidth = 1,
            edgecolor = "k"*20,
            palette  = "Spectral_r")

plt.xlabel("Attendance", fontsize=20)
plt.ylabel("City", fontsize=20)

ticks = np.arange(0, 100000, 10000)
labels = ["{}K".format(i//1000) for i in ticks]
plt.xticks(ticks, labels) 

for i,j in enumerate(" Average attendance  : "+np.around(ct_at["Attendance"][:20],0).astype(str)):
    ax.text(.7,i,j,fontsize=12)

plt.title("Average attendance by city" , fontsize = 25 , color = "orange")
plt.show()


# In[51]:


df_Cups["Winner"]=df_Cups["Winner"].replace("Germany FR","Germany")
df_Cups["Runners-Up"]=df_Cups["Runners-Up"].replace("Germany FR","Germany")
cou = df_Cups["Winner"].value_counts().reset_index()

plt.figure(figsize=(12,7))
sns.barplot("index","Winner",data=cou,palette="binary_r",
            linewidth=2,edgecolor="k"*len(cou))
plt.grid(True)
plt.ylabel("Number of times")
plt.xlabel("country")
plt.title("Teams with the most world cup victories")
plt.xticks(color="navy",fontsize=12)
plt.show()


# In[52]:


cou_w = cou.copy()
cou_w.columns = ["country","count"]
cou_w["type"] = "WINNER"

cou_r = df_Cups["Runners-Up"].value_counts().reset_index()
cou_r.columns = ["country","count"]
cou_r["type"] = "RUNNER - Up"

cou_t = pd.concat([cou_w,cou_r],axis=0)

plt.figure(figsize=(8,8))
sns.barplot("count","country",
            data=cou_t,
            hue="type",
            palette=["b","cyan"],
            linewidth=1,
            edgecolor="k"*len(cou_t))
plt.xlabel("Count", fontsize=20)
plt.ylabel("Country", fontsize=20)
plt.grid(True)
plt.legend(loc="center right",prop={"size":14})
plt.title("Final results by nation" , fontsize = 25 , color = "orange")
plt.show()


# In[53]:


third = df_Cups["Third"].value_counts().reset_index()
third.columns = ["team","count"]
third["type"] = "THIRD PLACE"

fourth = df_Cups["Fourth"].value_counts().reset_index()
fourth.columns = ["team","count"]
fourth["type"] = "FOURTH PLACE"

plcs = pd.concat([third,fourth],axis=0)

plt.figure(figsize=(8,8))
sns.barplot("count","team",data=plcs,hue="type",
           linewidth=1,edgecolor="k"*len(plcs),
           palette = "bone")
plt.grid(True)
plt.xticks(np.arange(0,4,1))
plt.xlabel("Count", fontsize=20)
plt.ylabel("Teams", fontsize=20)
plt.title(" World cup final result for third and fourth place" , fontsize = 25 , color = "orange")
plt.legend(loc="center right",prop={"size":12})
plt.show()


# In[54]:


df_Matches["Home Team Name"] = df_Matches["Home Team Name"].str.replace('rn">United Arab Emirates',"United Arab Emirates")
df_Matches["Home Team Name"] = df_Matches["Home Team Name"].str.replace("C�te d'Ivoire","Côte d’Ivoire")
df_Matches["Home Team Name"] = df_Matches["Home Team Name"].str.replace('rn">Republic of Ireland',"Republic of Ireland")
df_Matches["Home Team Name"] = df_Matches["Home Team Name"].str.replace('rn">Bosnia and Herzegovina',"Bosnia and Herzegovina")
df_Matches["Home Team Name"] = df_Matches["Home Team Name"].str.replace('rn">Serbia and Montenegro',"Serbia and Montenegro")
df_Matches["Home Team Name"] = df_Matches["Home Team Name"].str.replace('rn">Trinidad and Tobago',"Trinidad and Tobago")

df_Matches["Away Team Name"] = df_Matches["Away Team Name"].str.replace('rn">United Arab Emirates',"United Arab Emirates")
df_Matches["Away Team Name"] = df_Matches["Away Team Name"].str.replace("C�te d'Ivoire","Côte d’Ivoire")
df_Matches["Away Team Name"] = df_Matches["Away Team Name"].str.replace('rn">Republic of Ireland',"Republic of Ireland")
df_Matches["Away Team Name"] = df_Matches["Away Team Name"].str.replace('rn">Bosnia and Herzegovina',"Bosnia and Herzegovina")
df_Matches["Away Team Name"] = df_Matches["Away Team Name"].str.replace('rn">Serbia and Montenegro',"Serbia and Montenegro")
df_Matches["Away Team Name"] = df_Matches["Away Team Name"].str.replace('rn">Trinidad and Tobago',"Trinidad and Tobago")
df_Matches["Home Team Name"] = df_Matches["Home Team Name"].str.replace("Germany FR","Germany")
df_Matches["Away Team Name"] = df_Matches["Away Team Name"].str.replace("Germany FR","Germany")

ht = df_Matches["Home Team Name"].value_counts().reset_index()
ht.columns = ["team","matches"]
at = df_Matches["Away Team Name"].value_counts().reset_index()
at.columns = ["team","matches"]
mt = pd.concat([ht,at],axis=0)
mt = mt.groupby("team")["matches"].sum().reset_index().sort_values(by="matches",ascending=False)

plt.figure(figsize=(10,10))
ax = sns.barplot("matches","team",data=mt[:25],palette="gnuplot_r",
                 linewidth=1,edgecolor="k"*25)
plt.grid(True)
plt.xlabel("Teams", fontsize=20)
plt.ylabel("Matches", fontsize=20)
plt.title("Teams with the most matches", fontsize = 25 , color = "orange")
for i,j in enumerate("Matches played  : " + mt["matches"][:25].astype(str)):
    ax.text(.7,i,j,fontsize=13,color="cyan")


# In[55]:


hy = df_Matches[["Year","Home Team Name"]]
hy.columns = ["year","team"]
hy["type"] = "HOME TEAM"
ay = df_Matches[["Year","Away Team Name"]]
ay.columns = ["year","team"]
ay["type"] = "AWAY TEAM"

home_away = pd.concat([hy,ay],axis=0)
yt = home_away.groupby(["year","team"]).count().reset_index()
yt = yt["team"].value_counts().reset_index()

plt.figure(figsize=(10,8))
ax = sns.barplot("team","index",data=yt[:15],linewidth=1,edgecolor="k"*25)

for i,j in enumerate("Particpated " + yt["team"][:15].astype(str) + " times"):
    ax.text(.7,i,j,fontsize=14,color="k")
plt.xticks(np.arange(0,20,2))    
plt.xlabel("Team", fontsize=20)
plt.ylabel("Country", fontsize=20)    
plt.title("Teams with the most tournament participations" , fontsize = 25 , color = "orange")
plt.show()


# In[56]:


plt.figure(figsize=(12,13))
plt.subplot(211)
sns.distplot(df_Matches["Home Team Goals"],color="b",rug=True)
plt.xticks(np.arange(0,12,1))
plt.xlabel("Home Team Goals", fontsize=20)
plt.ylabel("Density", fontsize=20)    
plt.title("Distribution of Home Team Goals" , fontsize = 25 , color = "orange")
plt.show()


# In[57]:


plt.figure(figsize=(12,13))
plt.subplot(212)
sns.distplot(df_Matches["Away Team Goals"],color="r",rug=True)
plt.xticks(np.arange(0,12,1))
plt.xlabel("Away Team Goals", fontsize=20)
plt.ylabel("Density", fontsize=20)    
plt.title("Distribution of Away Team Goals" , fontsize = 25 , color = "orange")
plt.show()


# In[58]:


gh = df_Matches[["Year","Home Team Goals"]]
gh.columns = ["year","goals"]
gh["type"] = "Home Team Goals"

ga = df_Matches[["Year","Away Team Goals"]]
ga.columns = ["year","goals"]
ga["type"] = "Away Team Goals"

goals = pd.concat([ga,gh],axis=0)

plt.figure(figsize=(13,8))
sns.violinplot(goals["year"],goals["goals"],
               hue=goals["type"],split=True,inner="quart",palette="husl")

plt.title("Home and away goals by year")
plt.show()


# In[59]:


hhg = df_Matches[["Year",'Half-time Home Goals']]
hhg.columns = ["year","goals"]
hhg["type"] = 'first half home goals'

hag = df_Matches[["Year",'Half-time Home Goals']]
hag.columns = ["year","goals"]
hag["type"] = 'first half away goals'

half_time = pd.concat([hhg,hag],axis=0)

plt.figure(figsize=(13,8))
sns.violinplot(half_time["year"],half_time["goals"],hue=half_time["type"],
               split=True,inner="quart",palette="prism")
plt.title(" first half  home and away goals by year")
plt.show()


# In[60]:


shg = df_Matches[["Year",'Half-time Away Goals']]
shg.columns = ["year","goals"]
shg["type"] = 'Second Half Home Goals'

sag = df_Matches[["Year",'Half-time Away Goals']]
sag.columns = ["year","goals"]
sag["type"] = 'Second Half Away Goals'

s_half_time = pd.concat([shg,sag],axis=0)

plt.figure(figsize=(13,8))
sns.violinplot(s_half_time["year"],s_half_time["goals"],hue=s_half_time["type"],
               split=True,inner="quart",palette="prism")
plt.title("Second Half Home and Away Goals by year")
plt.show()


# In[61]:


def label(df_Matches):
    if df_Matches["Home Team Goals"] > df_Matches["Away Team Goals"]:
        return "Home team win"
    if df_Matches["Away Team Goals"] > df_Matches["Home Team Goals"]:
        return "Away team win"
    if df_Matches["Home Team Goals"] == df_Matches["Away Team Goals"]:
        return "DRAW"

df_Matches["outcome"] = df_Matches.apply(lambda df_Matches:label(df_Matches),axis=1)
plt.figure(figsize=(9,9))
df_Matches["outcome"].value_counts().plot.pie(autopct="%1.0f%%",fontsize =14,
                                           colors = sns.color_palette("husl"),
                                           wedgeprops={"linewidth":2,"edgecolor":"white"},
                                           shadow=True)
circ = plt.Circle((0,0),.7,color="white")
plt.gca().add_artist(circ)
plt.title("# Match outcomes by home and away teams" , fontsize = 25 , color = "orange")
plt.show()


# In[62]:


df_Matches[['Home Team Name','Home Team Goals', 'Away Team Goals', 'Away Team Name',"outcome"]]

def win_label(df_Matches):
    if df_Matches["Home Team Goals"] > df_Matches["Away Team Goals"]:
        return df_Matches["Home Team Name"]
    if df_Matches["Home Team Goals"] < df_Matches["Away Team Goals"]:
        return df_Matches["Away Team Name"]
    if df_Matches["Home Team Goals"] == df_Matches["Away Team Goals"]:
        return "DRAW"
    
def lost_label(df_Matches):
    if df_Matches["Home Team Goals"] < df_Matches["Away Team Goals"]:
        return df_Matches["Home Team Name"]
    if df_Matches["Home Team Goals"] > df_Matches["Away Team Goals"]:
        return df_Matches["Away Team Name"]
    if df_Matches["Home Team Goals"] == df_Matches["Away Team Goals"]:
        return "DRAW"
    
df_Matches["win_team"]  = df_Matches.apply(lambda df_Matches:win_label(df_Matches),axis=1)
df_Matches["lost_team"] = df_Matches.apply(lambda df_Matches:lost_label(df_Matches),axis=1)
lost = df_Matches["lost_team"].value_counts().reset_index()
win = df_Matches["win_team"].value_counts().reset_index()
wl  = win.merge(lost,left_on="index",right_on="index",how="left")
wl = wl[wl["index"] != "DRAW"]
wl.columns  = ["team","wins","loses"]


# In[63]:


wl1 = wl.copy()
wl1 = wl1.merge(mt,left_on="team",right_on="team",how="left")
wl1["draws"] = wl1["matches"]-(wl1["wins"]+wl1["loses"])
wl1.index = wl1.team
wl1 = wl1.sort_values(by="wins",ascending=True)
wl1[["wins","draws","loses"]].plot(kind="barh",
                                   stacked=True,
                                   figsize=(10,17),
                                  linewidth=1,edgecolor="k"*len(wl1))
plt.legend(loc="center right",prop={"size":20})
plt.xticks(np.arange(0,120,5))
plt.title("Match outcomes by countries" , fontsize = 25 , color = "orange")
plt.xlabel("Matches Played",fontsize = 20 )
plt.ylabel("Teams" , fontsize = 20 )
plt.show()


# In[64]:


cols = [ 'wins', 'loses', 'draws']
length = len(cols)

plt.figure(figsize=(8,18))
for i,j in itertools.zip_longest(cols,range(length)):
    plt.subplot(3,1,j+1)
    ax = sns.barplot(i,"team",data=wl1.sort_values(by=i,ascending=False)[:10],
                     linewidth=1,edgecolor="k"*10,palette="husl")
    for k,l in enumerate(wl1.sort_values(by=i,ascending=False)[:10][i]):
        ax.text(.7,k,l,fontsize=13)
    
    plt.grid(True)
    plt.title("Countries with maximum "+i,color="navy")


# In[65]:


tt_gl_h = df_Matches.groupby("Home Team Name")["Home Team Goals"].sum().reset_index()
tt_gl_h.columns = ["team","goals"]

tt_gl_a = df_Matches.groupby("Away Team Name")["Away Team Goals"].sum().reset_index()
tt_gl_a.columns = ["team","goals"]

total_goals = pd.concat([tt_gl_h,tt_gl_a],axis=0)
total_goals = total_goals.groupby("team")["goals"].sum().reset_index()
total_goals = total_goals.sort_values(by="goals",ascending =False)
total_goals["goals"] = total_goals["goals"].astype(int)

plt.figure(figsize=(10,12))
ax= sns.barplot("goals","team",data=total_goals[:20],palette="cool",
                linewidth=1,edgecolor="k"*20)

for i,j in enumerate("SCORED  " +total_goals["goals"][:20].astype(str) + "  GOALS"):
    ax.text(.7,i,j,fontsize = 10,color="k")

plt.title("Teams with highest fifa world cup goals")
plt.grid(True)


# In[66]:


df_Matches["total_goals"] = df_Matches["Home Team Goals"] + df_Matches["Away Team Goals"]
hig_gl = df_Matches.sort_values(by="total_goals",ascending =False)[:10][['Year', 'Datetime', 'Stage', 'Stadium', 'City', 'Home Team Name',
                                                                  'Home Team Goals', 'Away Team Goals', 'Away Team Name',"total_goals"]]
hig_gl["match"] = hig_gl["Home Team Name"] + " .Vs. " + hig_gl['Away Team Name']
hig_gl.index = hig_gl["match"]
hig_gl = hig_gl.sort_values(by="total_goals",ascending =True)
ax= hig_gl[["Home Team Goals","Away Team Goals"]].plot(kind="barh",stacked=True,
                                                       figsize=(10,8),linewidth=2,
                                                       edgecolor="w"*10)

plt.ylabel("home team vs away team",color="b")
plt.xlabel("goals",color="b")
plt.title("Highest total goals scored during a match ")

for i,j in enumerate("Date  : " + hig_gl["Datetime"]):
    ax.text(.7,i,j,color="w",fontsize=11)
    
plt.show()


# In[67]:


plt.figure(figsize=(13,8))
sns.boxplot(y=df_Matches["total_goals"],x=df_Matches["Year"])
plt.grid(True)
plt.title("Total goals scored during game by year")
plt.show()


# In[68]:


#TEAM COMPARATOR
df_Matches_played = mt.copy()
mat_new = df_Matches_played.merge(lost,left_on="team",right_on="index",how="left")
mat_new = mat_new.merge(win,left_on="team",right_on="index",how="left")
mat_new = mat_new[["team","matches","lost_team","win_team"]]
mat_new = mat_new.fillna(0) 
mat_new["win_team"] = mat_new["win_team"].astype(int)
mat_new["draws"]    = (mat_new["matches"]) - (mat_new["lost_team"] + mat_new["win_team"])
mat_new = mat_new.merge(total_goals,left_on="team",right_on="team",how="left")
mat_new = mat_new.rename(columns={"win_team":"wins","lost_team":"loses"})


# In[69]:


def team_compare(team1,team2):
    
    team_list = [team1,team2] 
    dat = mat_new[mat_new["team"].isin(team_list)]
    
    plt.figure(figsize=(12,8))
    cols   = ["matches","goals","wins","loses"]
    length = len(cols)
    
    for i,j in itertools.zip_longest(cols,range(length)):
        
        fig = plt.subplot(length/2,length/2,j+1)
        ax  = sns.barplot(dat[i],dat["team"],palette=["royalblue","r"],
                         linewidth=2,edgecolor="k"*len(team_list))
        plt.ylabel("")
        plt.yticks(fontsize=13)
        plt.grid(True,color="grey",alpha=.3)
        plt.title(i,color="navy",fontsize=15)
        plt.subplots_adjust(wspace = .3,hspace =.5)
        fig.set_facecolor("w")
        
        for k,l in enumerate(dat[i].values):
            ax.text(.7,k,l,weight = "bold",fontsize = 20)


# In[70]:


team_compare("Portugal","Argentina")


# In[71]:


team_compare("Italy","Spain")


# In[72]:


team_compare("Brazil","Germany")


# In[73]:


ref = df_Matches["Referee"].value_counts().reset_index()
ref = ref.sort_values(by="Referee",ascending=False)

plt.figure(figsize=(10,10))
sns.barplot("Referee","index",data=ref[:20],linewidth=1,edgecolor="k"*20)
plt.xlabel("count")
plt.ylabel("Refree name")
plt.grid(True)
plt.title("Referee's with most matches")
plt.show()


# In[74]:


mat_new["goals_per_match"] = mat_new["goals"] / mat_new["matches"]
cou_lst = mat_new.sort_values(by="wins",ascending=False)[:15]["team"].tolist()
cou_gpm = mat_new[mat_new["team"].isin(cou_lst)]
cou_gpm = cou_gpm.sort_values(by="goals_per_match",ascending=False)

plt.figure(figsize=(10,8))
ax = sns.barplot("goals_per_match","team",
                 linewidth=1,
                 edgecolor=["k"]*len(cou_gpm),
                 data=cou_gpm,
                 palette="Spectral")

for i,j in enumerate(np.round(cou_gpm["goals_per_match"],2).astype(str) + "  Goals per game"):
    ax.text(.1,i,j,color="k",weight = "bold")
    
plt.title("Goals per match for countries with highest wins")
plt.show()


# In[75]:


import networkx as nx 

def interactions(year,color):
    
    df  =  df_Matches[df_Matches["Year"] == year][["Home Team Name","Away Team Name"]]
    G   = nx.from_pandas_edgelist(df,"Home Team Name","Away Team Name")
    
    plt.figure(figsize=(10,9))
    
    nx.draw_kamada_kawai(G,with_labels = True,
                         node_size  = 2500,
                         node_color = color,
                         node_shape = "h",
                         linewidths  = 5 ,
                         font_size  = 13 ,
                         alpha=.8)
    
    plt.title("Interaction between teams :" + str(year) , fontsize =13 , color = "navy")


# In[76]:


interactions(2014,"b")


# In[77]:


interactions(1994,"red")


# In[78]:


interactions(1950,"green")


# In[79]:


interactions(1930,"brown")


# In[80]:


ysc = df_Matches[["Year","Stadium","City","MatchID"]]
cy  = df_Cups[["Year","Country"]]
ysc = ysc.merge(cy,left_on="Year",right_on="Year",how="left")
#ysc["Stadium"] = ysc["Stadium"].str.split(",").str[0]
ysc["std_cty"] = ysc["Stadium"] +" , "+ ysc["City"]

cnt_mat  =  ysc.groupby("Country")["MatchID"].nunique().reset_index()
cnt_mat  =  cnt_mat.sort_values(by= "MatchID",ascending=False)
cnt_mat


plt.figure(figsize=(10,8))

ax = sns.barplot("MatchID","Country",
                data=cnt_mat,
                linewidth=1,
                edgecolor="k"*cnt_mat["Country"].nunique())

for i,j in enumerate("Matches : " + cnt_mat["MatchID"].astype(str)):
    ax.text(.7 ,i,j)

plt.title("Total world cup matches played in each country")
plt.grid(True)
plt.show()


# In[81]:


ysc["Country_yr"] = ysc["Country"] + " - " + ysc["Year"].astype(str)


def stadium_country(country,color):
    
    dat2 = ysc[ysc["Country"] == country]
    
    plt.figure(figsize=(10,8))

    H = nx.from_pandas_edgelist(dat2,"Country","Stadium")

    nx.draw_kamada_kawai(H,
                         with_labels=True,
                         node_size  = 2500,
                         node_color = color,
                         node_shape = "s",
                         linewidths  = 7 ,
                         font_size  = 13 ,
                         alpha=.8)
                         


# In[82]:


stadium_country("Germany","orange")


# In[83]:


stadium_country("Mexico","green")


# In[84]:


stadium_country("USA","darkred")


# In[85]:


from wordcloud import WordCloud,STOPWORDS
# import io
# import base64
# from matplotlib import rc,animation
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.basemap import Basemap
# import folium
# import folium.plugins
# import os
wrds = df_Players["Player Name"].value_counts().keys()

wc = WordCloud(scale=5,max_words=1000,colormap="rainbow").generate(" ".join(wrds))
plt.figure(figsize=(13,14))
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.title("Player names")
plt.show()


# In[86]:


wrds1 = df_Players["Coach Name"].str.split("(").str[0].value_counts().keys()

wc1 = WordCloud(scale=5,max_words=1000,colormap="rainbow",background_color="black").generate(" ".join(wrds1))
plt.figure(figsize=(13,14))
plt.imshow(wc1,interpolation="bilinear")
plt.axis("off")
plt.title("Coach names")
plt.show()


# 
# Research question 1
# Which national teams are considered to be fovarites of the World Cup according to their goals?
# This research question is identifying Top participants of World Cup by their goals scored at several matches.Firstly,
# I realized that on my table there are data about goals that was scored at home and away by national teams.
# So, I need to sum this two values in order to find total goals of national teams.It means that from the table I need 4 columns:
#     "Home Team Name", "Home Team Goals", "Away Team Name", "Away Team Goals".

# In[87]:


scoredathome = dr.groupby('Home Team Name')['Home Team Goals'].sum().reset_index() ## counts how many goals have been scored by home teams
scoredathome.columns = ['country','home_goal'] ## divide it by columns 
scoredataway = dr.groupby('Away Team Name')['Away Team Goals'].sum().reset_index() ## counts how many goals have been scored by away teams
scoredataway.columns = ['country','away_goal'] ## divide it by columns
homeandaway = pd.merge(scoredathome,scoredataway,on='country') ## we bring goals scored at home and away
homeandaway ['Total goals scored'] = homeandaway ['home_goal'] + homeandaway ['away_goal'] ## we need to sum goals to find total goals
homeandaway = homeandaway.sort_values(by='Total goals scored', ascending=False) ## I used this code to sort data by descending order


# In[ ]:


homeandaway[:10].plot(kind='bar',y='Total goals scored',x='country',figsize=(15,7),color='b') ## I chose type of chart as barchart and take only 10 country which considered to be fovarites
plt.xlabel('Countries',size='15') ## Label X axis as Countries
plt.ylabel('Number of goals',size='15') ## Label Y axis as Nuber of goals
plt.title('Fovarites of the World Cup',color='r',size='25') ## Title of my barchart and as usual used red color to title 
plt.show()


# 
# Findings and comments to RQ1:
# According to barchart following national teams are considered to be favorites of World Cup: Brazil,Argentina,Germany,Italy,
#     France,Spain,Netherlands,Hungary and Uruguay. Brazil have won World Cup 5 times and footbal appeared in this country, 
#     that is why Brazil on the first place.Also, we can see big fluctuation after Brazil and gradualy decreases from Argentina
#     to Uruguay.Next interesting thing that I found that only South American and European countries are considered to be 
#     favorites.
# 
# I have found one thing to consider from this barchart. On the the table goals scored by Germany national team and Germany 
# FR are given sepately. I thought it is because of some historical changes of Germany. So, It will be better to sum data of
# Germany FR and Germany.

# 
# Research question 3
# How does the attendance of people has changed by development of World Cup?
# So, this question is about number of fans who attend the matches. Firstly, I had to find average number of fans who attended the matches by the year. So, on my dataset it shows number of attended fans of every football match by the year. That is why I found mean attendance and marked it on "Y" axis and respectively year on "X" axis. I already did data preparation part for this question by dropping NaN values from table. I need only two columns from the table, there are "Year" and "Attendance" columns.

# In[ ]:


q3  = dr.groupby("Year")["Attendance"].mean().reset_index() ## data preparation using groupby Year and mean of Attendance


# In[ ]:


plt.figure(figsize=(15,7)) ## I used figsize 15,7 as standard for all my charts
ax = sns.pointplot(q3["Year"],q3["Attendance"],color="b") ## using seaborn i plotted Year and Attendence graph  
ax.set_xlabel('Years',size='15') ## label x axis as Years
plt.legend(["Attendance of stadium"])
ax.set_ylabel('Attendance',size='15')## label y axis as Attendance
plt.title("Average attendance by year",color='r',size='25') ## Title of graph is Average attendance by year
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# 

# In[ ]:




