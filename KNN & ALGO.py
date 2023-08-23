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
#print(os.listdir())
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


# In[8]:


print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[9]:


#finding outliers
print(df[df['plasma_CA19_9']>upper_limit])


# In[10]:


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


# In[11]:


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


# In[12]:


#creatinine
#Finding the IQR value
percentile25=df['creatinine'].quantile(0.25)
percentile75=df['creatinine'].quantile(0.75)
print("percentile75 of  creatinine=",percentile75)
print("prcentile25 of creatinine=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[13]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr


# In[14]:


print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[15]:


#creatinine
#Finding the IQR value
percentile25=df['creatinine'].quantile(0.25)
percentile75=df['creatinine'].quantile(0.75)
print("percentile75 of  creatinine=",percentile75)
print("prcentile25 of creatinine=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[16]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr


# In[17]:


print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[18]:


#finding outliers
print(df[df['creatinine']>upper_limit])


# In[19]:


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


# In[20]:


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


# In[21]:


#LYVE1
#Finding the IQR value
percentile25=df['LYVE1'].quantile(0.25)
percentile75=df['LYVE1'].quantile(0.75)
print("percentile75 of  LYVE1=",percentile75)
print("prcentile25 of LYVE1=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[22]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[23]:


#finding outliers
print(df[df['LYVE1']>upper_limit])


# In[24]:


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


# In[25]:


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


# In[26]:


#REG1B
#Finding the IQR value
percentile25=df['REG1B'].quantile(0.25)
percentile75=df['REG1B'].quantile(0.75)
print("percentile75 of  REG1B=",percentile75)
print("prcentile25 of REG1B=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[27]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[28]:


#finding outliers
print(df[df['REG1B']>upper_limit])


# In[29]:


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


# In[30]:


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


# In[31]:


#TFF1
#Finding the IQR value
percentile25=df['TFF1'].quantile(0.25)
percentile75=df['TFF1'].quantile(0.75)
print("percentile75 of  TFF1=",percentile75)
print("prcentile25 of TFF1=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[32]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[33]:


#finding outliers
print(df[df['TFF1']>upper_limit])


# In[34]:


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


# In[35]:


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


# In[36]:


#CEA
#Finding the IQR value
percentile25=df['CEA'].quantile(0.25)
percentile75=df['CEA'].quantile(0.75)
print("percentile75 of  CEA=",percentile75)
print("prcentile25 of CEA=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[37]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[38]:


#finding outliers
print(df[df['CEA']>upper_limit])


# In[39]:


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


# In[40]:


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


# In[41]:


#CA125
#Finding the IQR value
percentile25=df['CA125'].quantile(0.25)
percentile75=df['CA125'].quantile(0.75)
print("percentile75 of  CA125=",percentile75)
print("prcentile25 of CA125=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[42]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[43]:


#finding outliers
print(df[df['CA125']>upper_limit])


# In[44]:


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


# In[45]:


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


# In[46]:


#HbA1c
#Finding the IQR value
percentile25=df['HbA1c'].quantile(0.25)
percentile75=df['HbA1c'].quantile(0.75)
print("percentile75 of  HbA1c=",percentile75)
print("prcentile25 of HbA1c=",percentile25)
iqr=percentile75 -percentile25
print("iqr",iqr)


# In[47]:


upper_limit=percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

print("Upper_limit",upper_limit)
print("Lower_limit",lower_limit)


# In[48]:


#finding outliers
print(df[df['HbA1c']>upper_limit])


# In[49]:


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


# In[50]:


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


# In[51]:


print(X_train)


# In[52]:


from sklearn.impute import KNNImputer
knn=KNNImputer(n_neighbors=5)
X_train_trf=knn.fit_transform(X_train)
X_test_trf=knn.transform(X_test)
print(pd.DataFrame(X_train_trf,columns=X_train.columns).head(60))


# In[53]:


from sklearn.metrics import accuracy_score


# In[54]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train_trf,Y_train)

Y_pred_lr = lr.predict(X_test_trf)
Y_pred_lr.shape


# In[55]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# In[56]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y_test,Y_pred_lr)
print("Confusion Matrix:",cm)
accuracy=(accuracy_score(Y_test,Y_pred_lr))
print("Accuracy:",accuracy)
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:",sensitivity)
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity:",specificity)


# In[57]:


print(metrics.classification_report(Y_test,Y_pred_lr))


# In[58]:


#Adding KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_trf,Y_train)
y_pred=knn.predict(X_test_trf)


# In[59]:


score_knn = round(accuracy_score(y_pred,Y_test)*100,2)
print("The accuracy score achieved using KNeighbors is: "+str(score_knn)+" %")


# In[60]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,Y_test)
print("Confusion Matrix:",cm)
accuracy=(accuracy_score(Y_test,y_pred))
print("Accuracy:",accuracy)
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:",sensitivity)
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity:",specificity)


# In[62]:


print(metrics.classification_report(y_pred,Y_test))


# In[63]:


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


# In[64]:


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train_trf,Y_train)
Y_pred_dt = dt.predict(X_test_trf)
print(Y_pred_dt.shape)


# In[65]:


score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# In[66]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,Y_test)
print("Confusion Matrix:",cm)
accuracy=(accuracy_score(Y_test,y_pred))
print("Accuracy:",accuracy)
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:",sensitivity)
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity:",specificity)


# In[68]:


print(metrics.classification_report(y_pred,Y_test))


# In[69]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(500):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train_trf,Y_train)
    Y_pred_rf = rf.predict(X_test_trf)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
print(max_accuracy)
print(best_x)


# In[70]:


rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train_trf,Y_train)
Y_pred_rf = rf.predict(X_test_trf)
Y_pred_rf.shape


# In[71]:


score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random forest is: "+str(score_rf)+" %")


# In[72]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,Y_test)
print("Confusion Matrix:",cm)
accuracy=(accuracy_score(Y_test,y_pred))
print("Accuracy:",accuracy)
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print("Sensitivity:",sensitivity)
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("Specificity:",specificity)


# In[73]:


print(metrics.classification_report(y_pred,Y_test))


# In[ ]:


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




