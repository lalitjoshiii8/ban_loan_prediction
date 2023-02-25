#!/usr/bin/env python
# coding: utf-8

# ----
# -----
# There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>LoanStatNew</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>loan_amnt</td>
#       <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>term</td>
#       <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>int_rate</td>
#       <td>Interest Rate on the loan</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>installment</td>
#       <td>The monthly payment owed by the borrower if the loan originates.</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>grade</td>
#       <td>LC assigned loan grade</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>sub_grade</td>
#       <td>LC assigned loan subgrade</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>emp_title</td>
#       <td>The job title supplied by the Borrower when applying for the loan.*</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>emp_length</td>
#       <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>home_ownership</td>
#       <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>annual_inc</td>
#       <td>The self-reported annual income provided by the borrower during registration.</td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td>verification_status</td>
#       <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td>issue_d</td>
#       <td>The month which the loan was funded</td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td>loan_status</td>
#       <td>Current status of the loan</td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td>purpose</td>
#       <td>A category provided by the borrower for the loan request.</td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td>title</td>
#       <td>The loan title provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td>zip_code</td>
#       <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td>addr_state</td>
#       <td>The state provided by the borrower in the loan application</td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td>dti</td>
#       <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>earliest_cr_line</td>
#       <td>The month the borrower's earliest reported credit line was opened</td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>open_acc</td>
#       <td>The number of open credit lines in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>20</th>
#       <td>pub_rec</td>
#       <td>Number of derogatory public records</td>
#     </tr>
#     <tr>
#       <th>21</th>
#       <td>revol_bal</td>
#       <td>Total credit revolving balance</td>
#     </tr>
#     <tr>
#       <th>22</th>
#       <td>revol_util</td>
#       <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
#     </tr>
#     <tr>
#       <th>23</th>
#       <td>total_acc</td>
#       <td>The total number of credit lines currently in the borrower's credit file</td>
#     </tr>
#     <tr>
#       <th>24</th>
#       <td>initial_list_status</td>
#       <td>The initial listing status of the loan. Possible values are – W, F</td>
#     </tr>
#     <tr>
#       <th>25</th>
#       <td>application_type</td>
#       <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
#     </tr>
#     <tr>
#       <th>26</th>
#       <td>mort_acc</td>
#       <td>Number of mortgage accounts.</td>
#     </tr>
#     <tr>
#       <th>27</th>
#       <td>pub_rec_bankruptcies</td>
#       <td>Number of public record bankruptcies</td>
#     </tr>
#   </tbody>
# </table>
# 
# ---
# ----

# In[1]:


import pandas as pd


# In[2]:


data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')


# In[3]:


print(data_info.loc['revol_util']['Description'])


# In[4]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[5]:


feat_info('mort_acc')


# ## Loading the data and other imports

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv('lending_club_loan_two.csv')


# In[8]:


df.info()


# In[9]:


sns.countplot(x='loan_status',data=df)


# In[10]:


plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)


# In[11]:


df.corr()


# In[12]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.ylim(10, 0)


# In[13]:


feat_info('installment')


# In[14]:


feat_info('loan_amnt')


# In[15]:


sns.scatterplot(x='installment',y='loan_amnt',data=df,)


# In[16]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# In[17]:


df.groupby('loan_status')['loan_amnt'].describe()


# In[18]:


sorted(df['grade'].unique())


# In[19]:


sorted(df['sub_grade'].unique())


# In[20]:


sns.countplot(x='grade',data=df,hue='loan_status')


# In[21]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )


# In[22]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')


# In[23]:


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')


# In[24]:


df['loan_status'].unique()


# In[25]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[26]:


df[['loan_repaid','loan_status']]


# In[27]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# In[28]:


df.head()


# In[29]:


len(df)


# In[30]:


df.isnull().sum()


# In[31]:


100* df.isnull().sum()/len(df)


# In[32]:


feat_info('emp_title')
print('\n')
feat_info('emp_length')


# In[33]:


df['emp_title'].nunique()


# In[34]:


df['emp_title'].value_counts()


# In[35]:


df = df.drop('emp_title',axis=1)


# In[36]:


sorted(df['emp_length'].dropna().unique())


# In[37]:


emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']


# In[38]:


plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order)


# In[39]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')


# In[40]:


emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']


# In[41]:


emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']


# In[42]:


emp_len = emp_co/emp_fp


# In[43]:


emp_len


# In[44]:


emp_len.plot(kind='bar')


# In[45]:


df = df.drop('emp_length',axis=1)


# In[46]:


df.isnull().sum()


# In[47]:


df['purpose'].head(10)


# In[48]:


df['title'].head(10)


# In[49]:


df = df.drop('title',axis=1)


# In[50]:


feat_info('mort_acc')


# In[51]:


df['mort_acc'].value_counts()


# In[52]:


print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()


# In[53]:


print("Mean of mort_acc column per total_acc")
df.groupby('total_acc').mean()['mort_acc']


# In[54]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[55]:


total_acc_avg[2.0]


# In[56]:


def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[57]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)


# In[58]:


df.isnull().sum()


# In[59]:


df = df.dropna()


# In[60]:


df.isnull().sum()


# In[61]:


df.select_dtypes(['object']).columns


# In[62]:


df['term'].value_counts()


# In[63]:


# Or just use .map()
df['term'] = df['term'].apply(lambda term: int(term[:3]))


# In[64]:


df = df.drop('grade',axis=1)


# In[65]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[66]:


df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)


# In[67]:


df.columns


# In[68]:


df.select_dtypes(['object']).columns


# In[69]:


# CODE HERE


# In[70]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[71]:


df['home_ownership'].value_counts()


# In[72]:


df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)


# In[73]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# In[74]:


dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[75]:


df = df.drop('issue_d',axis=1)


# In[76]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)


# In[77]:


df.select_dtypes(['object']).columns


# ## Train Test Split

# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


df = df.drop('loan_status',axis=1)


# In[80]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# In[81]:


# df = df.sample(frac=0.1,random_state=101)
print(len(df))


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[83]:


from sklearn.preprocessing import MinMaxScaler


# In[84]:


scaler = MinMaxScaler()


# In[85]:


X_train = scaler.fit_transform(X_train)


# In[86]:


X_test = scaler.transform(X_test)


# In[87]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


# In[90]:



model = Sequential()


# In[91]:


model = Sequential()




# input layer
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[92]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )


# In[93]:


from tensorflow.keras.models import load_model


# In[94]:


model.save('full_data_project_model.h5')  


# In[95]:


losses = pd.DataFrame(model.history.history)


# In[96]:


losses[['loss','val_loss']].plot()


# In[97]:


from sklearn.metrics import classification_report,confusion_matrix


# In[100]:


predictions = (model.predict(X_test) > 0.5).astype("int32") 


# In[101]:


print(classification_report(y_test,predictions))


# In[102]:


confusion_matrix(y_test,predictions)


# In[103]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[108]:


df.iloc[random_ind]['loan_repaid']

