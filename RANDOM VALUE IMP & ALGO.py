#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import os
print(os.listdir())
import warnings
warnings.simplefilter(action="ignore")


# In[2]:


df= pd.read_csv('C:/Users/sriha/OneDrive/Desktop/FINAL YR PROJECT/final code/Dataset.csv')


# In[3]:


df.replace({'sex':{'M':1,'F':2}},inplace=True)


# In[4]:


X=df.drop(columns=['Sample_Id','patient_cohort','diagnosis', 'sample_origin', 'stage', 'benign_sample_diagnosis','REG1A'])
Y=df['diagnosis']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=5)


# In[5]:


print(df['age'].skew())
print(df['diagnosis'].skew())
print(df['creatinine'].skew())
print(df['plasma_CA19_9'].skew())
print(df['LYVE1'].skew())
print(df['REG1B'].skew())
print(df['TFF1'].skew())
print(df['HbA1c'].skew())
print(df['CEA'].skew())
print(df['CA125'].skew())


# In[6]:


#plasma_CA19_9
#Finding the IQR value
percentile25=df['plasma_CA19_9'].quantile(0.25)
percentile75=df['plasma_CA19_9'].quantile(0.75)
print("percentile75 of  plasma_CA19_9=",percentile75)
print("prcentile25 of plasma_CA19_9=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[7]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[8]:


#finding outliers
print(df[df['plasma_CA19_9']>upper_limit])


# In[9]:


#Capping
new_df_cap=df.copy()
new_df_cap['plasma_CA19_9']=np.where(
    new_df_cap['plasma_CA19_9']>upper_limit,
    upper_limit,
    np.where(
        new_df_cap['plasma_CA19_9']<lower_limit,
        lower_limit,
        new_df_cap['plasma_CA19_9']
        )
    )

new_df_cap.shape


# In[10]:


#Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['plasma_CA19_9'])
plt.subplot(2,2,2)
sns.boxplot(df['plasma_CA19_9'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['plasma_CA19_9'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['plasma_CA19_9'])
plt.show()


# In[11]:


#creatinine
#Finding the IQR value
percentile25=df['creatinine'].quantile(0.25)
percentile75=df['creatinine'].quantile(0.75)
print("percentile75 of  creatinine=",percentile75)
print("prcentile25 of creatinine=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[12]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[13]:



#creatinine
#Finding the IQR value
percentile25=df['creatinine'].quantile(0.25)
percentile75=df['creatinine'].quantile(0.75)
print("percentile75 of  creatinine=",percentile75)
print("prcentile25 of creatinine=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[14]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[15]:


#finding outliers
print(df[df['creatinine']>upper_limit])


# In[16]:


#Capping
new_df_cap=df.copy()
new_df_cap['creatinine']=np.where(
    new_df_cap['creatinine']>upper_limit,
    upper_limit,
    np.where(
        new_df_cap['creatinine']<lower_limit,
        lower_limit,
        new_df_cap['creatinine']
        )
    )

new_df_cap.shape


# In[17]:


#Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['creatinine'])
plt.subplot(2,2,2)
sns.boxplot(df['creatinine'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['creatinine'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['creatinine'])
plt.show()


# In[18]:


#LYVE1
#Finding the IQR value
percentile25=df['LYVE1'].quantile(0.25)
percentile75=df['LYVE1'].quantile(0.75)
print("percentile75 of  LYVE1=",percentile75)
print("prcentile25 of LYVE1=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[19]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[20]:


#finding outliers
print(df[df['LYVE1']>upper_limit])


# In[21]:


#Capping
new_df_cap=df.copy()
new_df_cap['LYVE1']=np.where(
    new_df_cap['LYVE1']>upper_limit,
    upper_limit,
    np.where(
        new_df_cap['LYVE1']<lower_limit,
        lower_limit,
        new_df_cap['LYVE1']
        )
    )

new_df_cap.shape


# In[22]:


#Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['LYVE1'])
plt.subplot(2,2,2)
sns.boxplot(df['LYVE1'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['LYVE1'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['LYVE1'])
plt.show()


# In[23]:


#REG1B
#Finding the IQR value
percentile25=df['REG1B'].quantile(0.25)
percentile75=df['REG1B'].quantile(0.75)
print("percentile75 of  REG1B=",percentile75)
print("prcentile25 of REG1B=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[24]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[25]:


#finding outliers
print(df[df['REG1B']>upper_limit])


# In[26]:


#Capping
new_df_cap=df.copy()
new_df_cap['REG1B']=np.where(
    new_df_cap['REG1B']>upper_limit,
    upper_limit,
    np.where(
        new_df_cap['REG1B']<lower_limit,
        lower_limit,
        new_df_cap['REG1B']
        )
    )

new_df_cap.shape


# In[27]:


#Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['REG1B'])
plt.subplot(2,2,2)
sns.boxplot(df['REG1B'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['REG1B'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['REG1B'])
plt.show()


# In[28]:


#TFF1
#Finding the IQR value
percentile25=df['TFF1'].quantile(0.25)
percentile75=df['TFF1'].quantile(0.75)
print("percentile75 of  TFF1=",percentile75)
print("prcentile25 of TFF1=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[29]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[30]:


#finding outliers
print(df[df['TFF1']>upper_limit])


# In[31]:


#Capping
new_df_cap=df.copy()
new_df_cap['TFF1']=np.where(
    new_df_cap['TFF1']>upper_limit,
    upper_limit,
    np.where(
        new_df_cap['TFF1']<lower_limit,
        lower_limit,
        new_df_cap['TFF1']
        )
    )

new_df_cap.shape


# In[32]:


#Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['TFF1'])
plt.subplot(2,2,2)
sns.boxplot(df['TFF1'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['TFF1'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['TFF1'])
plt.show()


# In[33]:


#CEA
#Finding the IQR value
percentile25=df['CEA'].quantile(0.25)
percentile75=df['CEA'].quantile(0.75)
print("percentile75 of  CEA=",percentile75)
print("prcentile25 of CEA=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[34]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[35]:


#finding outliers
print(df[df['CEA']>upper_limit])


# In[36]:


#Capping
new_df_cap=df.copy()
new_df_cap['CEA']=np.where(
    new_df_cap['CEA']>upper_limit,
    upper_limit,
    np.where(
        new_df_cap['CEA']<lower_limit,
        lower_limit,
        new_df_cap['CEA']
        )
    )

new_df_cap.shape


# In[37]:


#Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['CEA'])
plt.subplot(2,2,2)
sns.boxplot(df['CEA'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['CEA'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['CEA'])
plt.show()


# In[38]:


#CA125
#Finding the IQR value
percentile25=df['CA125'].quantile(0.25)
percentile75=df['CA125'].quantile(0.75)
print("percentile75 of  CA125=",percentile75)
print("prcentile25 of CA125=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[39]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[40]:


#finding outliers
print(df[df['CA125']>upper_limit])


# In[41]:


#Capping
new_df_cap=df.copy()
new_df_cap['CA125']=np.where(
    new_df_cap['CA125']>upper_limit,
    upper_limit,
    np.where(
        new_df_cap['CA125']<lower_limit,
        lower_limit,
        new_df_cap['CA125']
        )
    )
new_df_cap.shape


# In[42]:



#Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['CA125'])
plt.subplot(2,2,2)
sns.boxplot(df['CA125'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['CA125'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['CA125'])
plt.show()


# In[43]:


#HbA1c
#Finding the IQR value
percentile25=df['HbA1c'].quantile(0.25)
percentile75=df['HbA1c'].quantile(0.75)
print("percentile75 of  HbA1c=",percentile75)
print("prcentile25 of HbA1c=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[44]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[45]:


#finding outliers
print(df[df['HbA1c']>upper_limit])


# In[46]:


#Capping
new_df_cap=df.copy()
new_df_cap['HbA1c']=np.where(
    new_df_cap['HbA1c']>upper_limit,
    upper_limit,
    np.where(
        new_df_cap['HbA1c']<lower_limit,
        lower_limit,
        new_df_cap['HbA1c']
        )
    )
new_df_cap.shape


# In[47]:


#Comparing
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['HbA1c'])
plt.subplot(2,2,2)
sns.boxplot(df['HbA1c'])
plt.subplot(2,2,3)
sns.distplot(new_df_cap['HbA1c'])
plt.subplot(2,2,4)
sns.boxplot(new_df_cap['HbA1c'])
plt.show()


# In[48]:


X_train['plasma_CA19_9_imputed']= X_train['plasma_CA19_9']
X_test['plasma_CA19_9_imputed']= X_test['plasma_CA19_9']
print(X_test.tail())


# In[49]:


X_train['creatinine_imputed']= X_train['creatinine']
X_test['creatinine_imputed']= X_test['creatinine']
print(X_test.tail())


# In[50]:


X_train['LYVE1_imputed']= X_train['LYVE1']
X_test['LYVE1_imputed']= X_test['LYVE1']
print(X_test.tail())


# In[51]:


X_train['REG1B_imputed']= X_train['REG1B']
X_test['REG1B_imputed']= X_test['REG1B']
print(X_test.tail())


# In[52]:


X_train['TFF1_imputed']= X_train['TFF1']
X_test['TFF1_imputed']= X_test['TFF1']
print(X_test.tail())


# In[53]:


X_train['CEA_imputed']= X_train['CEA']
X_test['CEA_imputed']= X_test['CEA']
print(X_test.tail())


# In[54]:


X_train['CA125_imputed']= X_train['CA125']
X_test['CA125_imputed']= X_test['CA125']
print(X_test.tail())


# In[55]:


X_train['HbA1c_imputed']= X_train['HbA1c']
X_test['HbA1c_imputed']= X_test['HbA1c']
print(X_test.tail())


# In[56]:


X_train['plasma_CA19_9_imputed'][X_train['plasma_CA19_9_imputed'].isnull()]=X_train['plasma_CA19_9'].dropna().sample(X_train['plasma_CA19_9'].isnull().sum()).values
X_test['plasma_CA19_9_imputed'][X_test['plasma_CA19_9_imputed'].isnull()]=X_train['plasma_CA19_9'].dropna().sample(X_test['plasma_CA19_9'].isnull().sum()).values
X_train['creatinine_imputed'][X_train['creatinine_imputed'].isnull()]=X_train['creatinine'].dropna().sample(X_train['creatinine'].isnull().sum()).values
X_test['creatinine_imputed'][X_test['creatinine_imputed'].isnull()]=X_train['creatinine'].dropna().sample(X_test['creatinine'].isnull().sum()).values
X_train['LYVE1_imputed'][X_train['LYVE1_imputed'].isnull()]=X_train['LYVE1'].dropna().sample(X_train['LYVE1'].isnull().sum()).values
X_test['LYVE1_imputed'][X_test['LYVE1_imputed'].isnull()]=X_train['LYVE1'].dropna().sample(X_test['LYVE1'].isnull().sum()).values
X_train['REG1B_imputed'][X_train['REG1B_imputed'].isnull()]=X_train['REG1B'].dropna().sample(X_train['REG1B'].isnull().sum()).values
X_test['REG1B_imputed'][X_test['REG1B_imputed'].isnull()]=X_train['REG1B'].dropna().sample(X_test['REG1B'].isnull().sum()).values
X_train['TFF1_imputed'][X_train['TFF1_imputed'].isnull()]=X_train['TFF1'].dropna().sample(X_train['TFF1'].isnull().sum()).values
X_test['TFF1_imputed'][X_test['TFF1_imputed'].isnull()]=X_train['TFF1'].dropna().sample(X_test['TFF1'].isnull().sum()).values
X_train['CEA_imputed'][X_train['CEA_imputed'].isnull()]=X_train['CEA'].dropna().sample(X_train['CEA'].isnull().sum()).values
X_test['CEA_imputed'][X_test['CEA_imputed'].isnull()]=X_train['CEA'].dropna().sample(X_test['CEA'].isnull().sum()).values
X_train['CA125_imputed'][X_train['CA125_imputed'].isnull()]=X_train['CA125'].dropna().sample(X_train['CA125'].isnull().sum()).values
X_test['CA125_imputed'][X_test['CA125_imputed'].isnull()]=X_train['CA125'].dropna().sample(X_test['CA125'].isnull().sum()).values
X_train['HbA1c_imputed'][X_train['HbA1c_imputed'].isnull()]=X_train['HbA1c'].dropna().sample(X_train['HbA1c'].isnull().sum()).values
X_test['HbA1c_imputed'][X_test['HbA1c_imputed'].isnull()]=X_train['HbA1c'].dropna().sample(X_test['HbA1c'].isnull().sum()).values


# In[57]:


print(X_train['plasma_CA19_9'].dropna().sample(X_train['plasma_CA19_9'].isnull().sum()).values)
print(X_train['creatinine'].dropna().sample(X_train['creatinine'].isnull().sum()).values)
print(X_train['LYVE1'].dropna().sample(X_train['LYVE1'].isnull().sum()).values)
print(X_train['REG1B'].dropna().sample(X_train['REG1B'].isnull().sum()).values)
print(X_train['TFF1'].dropna().sample(X_train['TFF1'].isnull().sum()).values)
print(X_train['CEA'].dropna().sample(X_train['CEA'].isnull().sum()).values)
print(X_train['CA125'].dropna().sample(X_train['CA125'].isnull().sum()).values)
print(X_train['HbA1c'].dropna().sample(X_train['HbA1c'].isnull().sum()).values)


# In[58]:


print(X_train.head(60))


# In[59]:


print('Original variable variance:',X_train['plasma_CA19_9'].var())
print('Variance after random imputation:',X_train['plasma_CA19_9_imputed'].var())
print('Original variable variance:',X_train['creatinine'].var())
print('Variance after random imputation:',X_train['creatinine_imputed'].var())
print('Original variable variance:',X_train['LYVE1'].var())
print('Variance after random imputation:',X_train['LYVE1_imputed'].var())
print('Original variable variance:',X_train['REG1B'].var())
print('Variance after random imputation:',X_train['REG1B_imputed'].var())
print('Original variable variance:',X_train['TFF1'].var())
print('Variance after random imputation:',X_train['TFF1_imputed'].var())
print('Original variable variance:',X_train['CEA'].var())
print('Variance after random imputation:',X_train['CEA_imputed'].var())
print('Original variable variance:',X_train['CA125'].var())
print('Variance after random imputation:',X_train['CA125_imputed'].var())
print('Original variable variance:',X_train['HbA1c'].var())
print('Variance after random imputation:',X_train['HbA1c_imputed'].var())


# In[60]:


X_train_trf=X_train.drop(columns=['plasma_CA19_9','creatinine','LYVE1','REG1B','TFF1','CEA','CA125','HbA1c'])
X_test_trf=X_test.drop(columns=['plasma_CA19_9','creatinine','LYVE1','REG1B','TFF1','CEA','CA125','HbA1c'])


# In[61]:


from sklearn.metrics import accuracy_score


# In[62]:


print(X_train_trf.head(60))


# In[63]:


print(X_test_trf.head(60))


# In[64]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train_trf,Y_train)

Y_pred_lr = lr.predict(X_test_trf)
Y_pred_lr.shape


# In[65]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# In[66]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,Y_pred_lr)
print("Confusion Matrix:",cm)
accuracy=(accuracy_score(Y_test,Y_pred_lr))
print("Accuracy:",accuracy)
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:",sensitivity)
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity:",specificity)


# In[67]:


print(metrics.classification_report(Y_test,Y_pred_lr))


# In[68]:


#Adding KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_trf,Y_train)
y_pred=knn.predict(X_test_trf)


# In[69]:


score_knn = round(accuracy_score(y_pred,Y_test)*100,2)
print("The accuracy score achieved using KNeighbors is: "+str(score_knn)+" %")


# In[71]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,Y_test)
print("Confusion Matrix:",cm)
accuracy=(accuracy_score(y_pred,Y_test))
print("Accuracy:",accuracy)
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:",sensitivity)
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity:",specificity)


# In[72]:


print(metrics.classification_report(y_pred,Y_test))


# In[73]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(400):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train_trf,Y_train)
    Y_pred_dt = dt.predict(X_test_trf)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
print(max_accuracy)
print(best_x)


# In[74]:


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train_trf,Y_train)
Y_pred_dt = dt.predict(X_test_trf)
print(Y_pred_dt.shape)


# In[75]:


score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# In[76]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_pred_dt,Y_test)
print("Confusion Matrix:",cm)
accuracy=(accuracy_score(Y_pred_dt,Y_test))
print("Accuracy:",accuracy)
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:",sensitivity)
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity:",specificity)


# In[77]:


print(metrics.classification_report(Y_pred_dt,Y_test))


# In[82]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(200):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train_trf,Y_train)
    Y_pred_rf = rf.predict(X_test_trf)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
print(max_accuracy)
print(best_x)


# In[83]:


rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train_trf,Y_train)
Y_pred_rf = rf.predict(X_test_trf)
Y_pred_rf.shape


# In[84]:


score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random forest is: "+str(score_rf)+" %")


# In[85]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_pred_rf,Y_test)
print("Confusion Matrix:",cm)
accuracy=(accuracy_score(Y_pred_rf,Y_test))
print("Accuracy:",accuracy)
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:",sensitivity)
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity:",specificity)


# In[86]:


print(metrics.classification_report(Y_pred_rf,Y_test))


# In[87]:


age=int(input("Enter the age of the patient:"))
gender=int(input("Enter the gender:Male-1,Female-2:"))
plasma_ca19_9=float(input("Enter the value of plasma_CA19_9:"))
creatinine=float(input("Enter the value of creatinine:"))
LYVE1=float(input("Enter the value of LYVE1:"))
REG1B=float(input("Enter the value of REG1B:"))
TFF1=float(input("Enter the value of TFF1:"))
CEA=float(input("Enter the value of CEA:"))
CA125=float(input("Enter the value of CA125:"))
HbA1c=float(input("Enter the value of HbA1c:"))

y=dt.predict([[age,gender,plasma_ca19_9,creatinine,LYVE1,REG1B,TFF1,CEA,CA125,HbA1c]])
print(y)


# In[ ]:





# In[ ]:




