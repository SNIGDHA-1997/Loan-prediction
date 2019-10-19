
# coding: utf-8

# In[140]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings                        # To ignore any warnings 
warnings.filterwarnings("ignore")


# In[141]:

train=pd.read_csv(r"C:\Users\snigd\Downloads\train_loan.csv")
test=pd.read_csv(r"C:\Users\snigd\Downloads\test_loan.csv")


# In[142]:

train_original=train.copy() 
test_original=test.copy()


# In[143]:

train.columns


# In[144]:

test.columns


# In[145]:

train.dtypes


# In[146]:

train.shape,test.shape


# In[147]:

plt.figure(1)
plt.subplot(111)
train['Loan_Status'].value_counts(normalize=True).plot.bar(figsize=(3,3),title="Loan Status")
plt.show()


# In[148]:

plt.figure(1)
plt.subplot(221) 
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 
plt.subplot(222) 
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
plt.subplot(223) 
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 
plt.subplot(224) 
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 
plt.show()


# In[149]:

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(15,3),title="Dependants")

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(figsize=(15,3),title="Education")

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(15,3),title="Property_Area")

plt.show()


# In[150]:

plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# In[151]:

train.boxplot(column='ApplicantIncome', by = 'Education')

plt.show()


# In[152]:

train['Gender'].fillna(train['Gender'].mode(), inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[153]:

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)


# In[154]:

train.isnull().sum()


# In[155]:

train['LoanAmount'].fillna(train['LoanAmount'].mean(),inplace=True)


# In[156]:

train.isnull().sum()


# In[157]:

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)


# In[158]:

train['LoanAmount_log'] = np.log(train['LoanAmount']) 

test['LoanAmount_log'] = np.log(test['LoanAmount'])


# In[159]:

train['LoanAmount_log'].hist(color="black") 

plt.show()


# In[160]:


sns.distplot(train['LoanAmount_log'],color="violet")
plt.show()


# In[161]:




# In[162]:




# In[163]:




# In[164]:




# In[ ]:




# In[165]:




# In[166]:




# In[167]:

train.tail()


# In[168]:

train.loc[train.Dependents=='3+','Dependents']= 4


# In[169]:




# In[170]:




# In[ ]:




# In[171]:




# In[173]:




# In[ ]:




# In[175]:

train.loc[train.Loan_Status=='N','Loan_Status']= 0
train.loc[train.Loan_Status=='Y','Loan_Status']=1
train.loc[train.Gender=='Male','Gender']= 0
train.loc[train.Gender=='Female','Gender']=1
train.loc[train.Married=='No','Married']= 0
train.loc[train.Married=='Yes','Married']=1
train.loc[train.Education=='Graduate','Education']= 0
train.loc[train.Education=='Not Graduate','Education']=1
train.loc[train.Self_Employed=='No','Self_Employed']= 0
train.loc[train.Self_Employed=='Yes','Self_Employed']=1


# In[183]:

X=train.drop(['Loan_ID','Loan_Status','Property_Area'],axis=1)


# In[179]:

y=train['Loan_Status']


# In[184]:

X


# In[185]:

y


# In[186]:

from sklearn.cross_validation import train_test_split


# In[187]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[190]:

X_train.loc[X_train.Gender=='Male','Gender']= 0
X_train.loc[X_train.Gender=='Female','Gender']=1


# In[193]:

X['Gender'].fillna(X['Gender'].mode()[0], inplace=True) 


# In[196]:

X_train['Gender'].fillna(X_train['Gender'].mode()[0], inplace=True) 


# In[197]:

X_train


# In[198]:

X_test['Gender'].fillna(X_test['Gender'].mode()[0], inplace=True) 


# In[199]:

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[200]:

prediction= logmodel.predict(X_test)


# In[ ]:




# In[203]:

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))


# In[204]:

test.tail()


# In[211]:


test.loc[test.Gender=='Male','Gender']= 0
test.loc[test.Gender=='Female','Gender']=1
test.loc[test.Married=='No','Married']= 0
test.loc[test.Married=='Yes','Married']=1
test.loc[test.Education=='Graduate','Education']= 0
test.loc[test.Education=='Not Graduate','Education']=1
test.loc[test.Self_Employed=='No','Self_Employed']= 0
test.loc[test.Self_Employed=='Yes','Self_Employed']=1


# In[212]:

test.head()


# In[213]:

test.loc[test.Dependents=='3+','Dependents']=test.Dependents.mode()


# In[214]:

X_data_test= test.drop(['Loan_ID','Property_Area'],axis=1)


# In[215]:

X_data_test.head()


# In[217]:

X_data_test['Gender'].fillna(X_data_test['Gender'].mode(),inplace=True)


# In[220]:

X_data_test['Dependents'].fillna(4,inplace=True)


# In[221]:

X_data_test


# In[222]:

test['Loan_Status']= logmodel.predict(X_data_test)


# In[223]:

data_frame=test[['Loan_ID','Loan_Status']]


# In[227]:

data_frame.loc[data_frame.Loan_Status==0,'Loan_Status']='N'
data_frame.loc[data_frame.Loan_Status==1,'Loan_Status']='Y'


# In[228]:

data_frame.head()


# In[ ]:



