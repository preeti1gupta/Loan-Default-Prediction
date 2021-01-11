#!/usr/bin/env python
# coding: utf-8

# In[1]:


import imblearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_curve, auc, plot_roc_curve
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression


# In[2]:


data = pd.read_csv('C:\\projects data sciencxe profile\\Prediction of Credit Risk Modelling\\loans-3.csv')


# # EDA

# In[4]:


data.head()


# In[5]:


data.info()


# In[35]:


data.isnull().sum()


# In[6]:


data.describe()


# In[7]:


sns.heatmap(data.corr(), annot=True, square=True) 
plt.show()


# In[8]:


sns.pairplot(data, hue='status')
plt.show()


# In[10]:


sns.countplot(x = data.status)
plt.show()


# In[11]:


sns.countplot(x = data.salary)
plt.show()


# In[12]:


sns.countplot(x = data.civil)
plt.show()


# In[14]:


sns.countplot(x = data.Age)
plt.show()


# In[20]:


sns.countplot(x=data.salary, hue=data.education) 
plt.show()


# In[22]:


sns.countplot(x=data.gender, hue=data.salary) 
plt.show()


# In[24]:


sns.countplot(x=data.status, hue=data.married_status) 
plt.show()


# In[25]:


g = sns.FacetGrid(data, col='status', hue="education")
plt.grid(True)
g.map(sns.countplot, "education", alpha=1)
g.add_legend()
plt.grid((False))


# In[26]:


data.status.describe()


# In[28]:


sns.violinplot(x=data.status)
plt.grid(True)


# In[50]:


data_train=data.sample(frac=0.70,random_state=200)
data_test=data.drop(data_train.index)


# In[52]:


import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[57]:


model1=smf.glm("status~C(education)+C(gender)+salary+Age+civil+appcount+phonegrade+simstrength",data=data_train, 
             family=sm.families.Binomial()).fit()


# In[58]:


print(model1.summary())


# In[59]:


model2=smf.glm("status~salary+Age+civil+appcount+phonegrade+simstrength", data=data_train,
              family=sm.families.Binomial()).fit()


# In[60]:


print(model2.summary())


# In[61]:


import sklearn.metrics as metrics


# In[64]:


y_true=data_test['status']
y_pred=model2.predict(data_test)


# In[65]:


y_pred.head()


# # Confusion Matrix

# In[66]:


y_true=data_test['status']
y_pred=model2.predict(data_test).map(lambda x:1 if x>0.5 else 0)
metrics.confusion_matrix(y_true,y_pred)


# # ROC

# In[67]:


y_score=model2.predict(data_test)
fpr,tpr,thresholds=metrics.roc_curve(y_true,y_score)
x, y=np.arange(0,1.1,0.1),np.arange(0,1.1,0.1)


# In[68]:


plt.plot(fpr,tpr,"-")
plt.plot(x,y,'b--')


# # AUC CURVE

# In[69]:


metrics.roc_auc_score(y_true,y_score)


# In[71]:


#Gains
data_test['prob']=model2.predict(data_test)


# In[72]:


data_test['prob'].head()


# In[73]:


data_test['prob_deciles']=pd.qcut(data_test['prob'],q=10)


# In[74]:


data_test.head()


# In[75]:


data_test.sort_values('prob',ascending=False).head()


# In[77]:


gains=data_test.groupby("prob_deciles",as_index=False)['status'].agg(['sum','count']).reset_index().sort_values("prob_deciles",
                 ascending=False)


# In[79]:


gains.columns=["Deciles","TotalEvents","NumberObs"]


# In[80]:


gains["PercEvents"]=gains['TotalEvents']/gains['TotalEvents'].sum()


# In[81]:


gains


# In[85]:


data_test.sort_values("prob",ascending=False)[['Age']].head(90)

