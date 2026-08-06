#!/usr/bin/env python
# coding: utf-8

# # NAME: Prabhkirat Singh Marwah
# # SEC: B
# # BATCH: B4
# # ROLL NO: 47
# # ML LAB : Practical 1

# In[ ]:





# In[11]:


# Step1: Load Dataset
df = pd.read_csv("Titanic-Dataset.xls")

#Display first 5 rows
df.head()


# In[12]:


df.tail()


# In[13]:


print(df.shape)


# In[14]:


df.columns


# In[15]:


df.info()


# In[16]:


df.describe()


# In[17]:


df.isnull().sum()


# In[18]:


plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(),cbar=False,cmap="viridis")
plt.title("Missing Values")
plt.show()


# In[19]:


#Step 3:Handle missing Values
df["Age"].fillna(df["Age"].median(),inplace=True)


# In[20]:


df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)


# In[21]:


df.isnull().sum()


# In[22]:


df.drop("Cabin",axis=1, inplace=True)


# In[23]:


df.isnull().sum()


# In[24]:


df.duplicated().sum()


# In[25]:


sns.boxplot(x=df["Age"])
plt.show()


# In[26]:


sns.boxplot(x=df["Fare"])
plt.show()


# In[27]:


#Remove Outliers using IQR Method
Q1=df["Fare"].quantile(0.25)
Q3=df["Fare"].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

df=df[(df["Fare"]>=lower) & (df["Fare"]<=upper)]


# In[28]:


sns.boxplot(x=df["Fare"])
plt.show()


# In[29]:


#Remove Outliers using IQR Method
Q1=df["Age"].quantile(0.25)
Q3=df["Age"].quantile(0.75)

IQR=Q3-Q1

lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

df=df[(df["Age"]>=lower) & (df["Age"]<=upper)]


# In[30]:


sns.boxplot(x=df["Age"])
plt.show()


# # Data Encoding

# In[31]:


df["Sex"] = df["Sex"].replace("male",0)
df["Sex"] = df["Sex"].replace("female",1)


# In[32]:


df.head()


# In[33]:


df["Embarked"] = df["Embarked"].replace("S",0)
df["Embarked"] = df["Embarked"].replace("Q",1)
df["Embarked"] = df["Embarked"].replace("C",2)


# In[34]:


df.head()


# In[35]:


df['Embarked'].unique()


# In[ ]:





# # univariate analysis

# In[37]:


plt.figure(figsize=(6,4))
sns.histplot(df['Age'],bins=20,kde=True)

plt.title('Age Dsitribution')
plt.show()


# In[40]:


plt.figure(figsize=(6,4))
sns.histplot(df['Fare'],bins=20,kde=True)

plt.title('Fare Dsitribution')
plt.show()


# In[39]:


sns.countplot(x="Sex",data=df)
plt.title("Gender Count")
plt.show()


# In[42]:


sns.countplot(x="Pclass",data=df)
plt.title("Gender Count")
plt.show()


# In[43]:


sns.countplot(x="Embarked",data=df)
plt.title("Gender Count")
plt.show()


# In[ ]:





# # bivariate analysis

# In[44]:


sns.countplot(x='Sex', hue='Survived', data=df)

plt.title("Gender vs Survival")
plt.show()


# In[45]:


sns.countplot(x='Pclass', hue='Survived', data=df)

plt.title("Passenger vs Survival")
plt.show()


# In[47]:


sns.countplot(x='Embarked', hue='Survived', data=df)

plt.title("Embarked vs Survival")
plt.show()


# In[46]:


sns.scatterplot(x='Age', y='Fare', data=df)

plt.title('Age vs Fare')
plt.show()


# In[ ]:





# # Multivariate Analysis

# In[56]:


plt.figure(figsize=(10,8))

numeric_df=df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap='inferno')
plt.show()
## colors are inferno , magma, plasmo,cividis


# In[ ]:





# In[58]:


#feature scaling
X=df.drop("Survived", axis=1)
y=df["Survived"]


# In[59]:


X=X.drop(["PassengerId",'Name','Ticket'], axis=1)


# In[64]:


#standard scaling
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X[['Age','Fare']] = scaler.fit_transform(X[['Age','Fare']])


# In[66]:


X.head(15)


# In[69]:


#train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20, random_state=42,train_size=0.80)


# In[71]:


print('Training data: ',X_train.shape)

print("Testing data : ",X_test.shape)


# In[72]:


print(X_train.head())


# In[ ]:




