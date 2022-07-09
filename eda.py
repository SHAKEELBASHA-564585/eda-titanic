
import pandas as pd
import seaborn as sns

df = pd.read_csv("Titanic-Train-Data.csv")

df.head()



if df["Pclass"].isnull().sum() == 0:
    print("Pclass has no missing values")
else:
    print("Pclass have missing values")

if df["Survived"].isnull().sum() == 0:
    print("Survived has no missing values")
else:
    print("Survived have missing values")

if df["Name"].isnull().sum() == 0:
    print("Name has no missing values")
else:
    print("Name have missing values")

if df["Sex"].isnull().sum() == 0:
    print("Sex has no missing values")
else:
    print("Sex have missing values")

if df["Age"].isnull().sum() == 0:
    print("Age has no missing values")
else:
    print("Age have missing values")

if df["SibSp"].isnull().sum() == 0:
    print("SibSp has no missing values")
else:
    print("SibSp have missing values")

if df["Parch"].isnull().sum() == 0:
    print("Parch has no missing values")
else:
    print("Parch have missing values")

if df["Ticket"].isnull().sum() == 0:
    print("Ticket has no missing values")
else:
    print("Ticket have missing values")

if df["Fare"].isnull().sum() == 0:
    print("Fare has no missing values")
else:
    print("Fare have missing values")

if df["Cabin"].isnull().sum() == 0:
    print("Cabin has no missing values")
else:
    print("Cabin have missing values")

if df["Embarked"].isnull().sum() == 0:
    print("Embarked has no missing values")
else:
    print("Embarked have missing values")

## feature engineering
# title extraction
for i in df.index:
    print(df["Name"][i].split(", ")[1].split(". ")[0])

# ticket first letters
for i in df.index:
    print(df["Ticket"][i][0])

# cabin first letters
#df["Cabin"].isnull().sum()


# In[129]:


# age imputation
df["Age"] = df["Age"].fillna(df["Age"].mean())


# In[131]:


# imputation for cabin
df["Cabin"]  = df["Cabin"].fillna(df["Cabin"].mode()[2])


# In[66]:


# embark imputation
df["Embarked"] = df["Embarked"].fillna(str(df["Embarked"].mode()).split("    ")[1].split("\n")[0])


# In[72]:


# encoding sex column
from sklearn.preprocessing import LabelEncoder

Sex_encoder = LabelEncoder()
Sex_encoder.fit(df['Sex'])
Sex_values = Sex_encoder.transform(df['Sex'])

print("Before Encoding:", list(df['Sex'][-10:]))
print("After Encoding:", Sex_values[-10:])
print("The inverse from the encoding result:", Sex_encoder.inverse_transform(Sex_values[-10:]))


# In[83]:


# family size
df["family size"] = df["SibSp"] + df["Parch"]
df["family size"]


# In[94]:


df.head()


# In[96]:





# In[106]:


#one hot encoding for sex column
y = pd.get_dummies(df.Sex, prefix='Sex')
df[["Sex_female","Sex_male"]] = y
y


# In[107]:


#one hot encoding for sex embarked
y = pd.get_dummies(df.Embarked, prefix='Embarked')
df[["Embarked_C","Embarked_Q","Embarked_S"]] = y
y


# In[108]:


df.head()


# In[109]:


# heatmap for missing values


# In[110]:


df1 = pd.read_csv("Titanic-Train-Data.csv")


# In[114]:

import seaborn as sns

sns.heatmap(df1.isnull(),cbar=False)

#!/usr/bin/env python
# coding: utf-8

# In[81]:


# sex vs age bar graph
import pandas as pd


# In[82]:


df = pd.read_csv("Titanic-Train-Data.csv")
df["Age"] = df["Age"].fillna(df["Age"].mean())


# In[83]:


data = []
import math
list_ages = []
for i  in df.index:
    ceil_value = round(df["Age"][i]) - (round(df["Age"][i]) % 10)
    add_value = (10 - (round(df["Age"][i]) % 10)) + (round(df["Age"][i]) % 10)
    floor_value = ceil_value + add_value
    data.append([f"{ceil_value}-{floor_value}",df["Sex"][i]])


# In[84]:


df1 = pd.DataFrame(data, columns=["Age","Gender"])
pd.crosstab(df1['Age'],df1['Gender']).plot.bar()


# In[91]:


data1 = []
import math
list_ages = []
for i  in df.index:
    ceil_value = round(df["Age"][i]) - (round(df["Age"][i]) % 10)
    add_value = (10 - (round(df["Age"][i]) % 10)) + (round(df["Age"][i]) % 10)
    floor_value = ceil_value + add_value
    data1.append([f"{ceil_value}-{floor_value}",df["Pclass"][i]])


# In[99]:



df2 = pd.DataFrame(data1, columns=["age","Pclass"])
pd.crosstab(df2['age'],df2['Pclass']).plot.bar()


# In[102]:


# age violinplot
from matplotlib import pyplot
import seaborn as sns
fig, ax = pyplot.subplots(figsize =(9, 7))
sns.violinplot( ax = ax, y = df["Age"] )


# In[103]:


# Pclass violinplot
fig, ax = pyplot.subplots(figsize =(9, 7))
sns.violinplot( ax = ax, y = df["Pclass"] )


# In[106]:


# sex violinplot
from sklearn.preprocessing import LabelEncoder

Sex_encoder = LabelEncoder()
Sex_encoder.fit(df['Sex'])
Sex_values = Sex_encoder.transform(df['Sex'])
fig, ax = pyplot.subplots(figsize =(9, 7))
sns.violinplot( ax = ax, y = Sex_values )
