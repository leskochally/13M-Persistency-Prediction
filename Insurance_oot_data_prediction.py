import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Preprocessing Steps done same as the training dataset
dataset=pd.read_csv('Assignment_OOT_Data.csv')
dataset.head()

dataset.isnull().sum()
#Dropping the Policy Identifier
dataset = dataset.drop(['VAR1'],axis=1)


dataset.dtypes
dataset["VAR2"] = pd.to_numeric(dataset["VAR2"],errors = "coerce")
#distribution plot to see outliers
sns.distplot(dataset['VAR2'].dropna())
dataset.boxplot("VAR2")
dataset['VAR2'].describe()

dataset[dataset.VAR2.notnull()]

#replaced the values of Agent persistency from 0 to 1 and also filled na values with 0
for i in range(len(dataset)):
    if(dataset.VAR2[i]>1):
        dataset.VAR2[i]=dataset.VAR2[i]/100.0
        
dataset['VAR2']=dataset.VAR2.fillna(0)
dataset = dataset.drop(["VAR3"],axis=1)
dataset['VAR4'] = dataset.VAR4.fillna('No Response')


dataset = dataset.dropna(how='any', subset=['VAR5', 'VAR6'])
def impute_nan(df,variable):
    dataset[variable+"_random"]=dataset[variable]
    
    random_sample=dataset[variable].dropna().sample(dataset[variable].isnull().sum(),random_state=0)
    
    random_sample.index=dataset[dataset[variable].isnull()].index
    dataset.loc[dataset[variable].isnull(),variable+'_random']=random_sample
impute_nan(dataset,"VAR8")
dataset = dataset.drop(["VAR8"],axis = 1)
dataset=dataset.rename(columns = {'VAR8_random':'VAR8'})


dataset.isnull().sum()

#Replaced null values of VAR 10 with highest frquency Category
dataset.VAR10.value_counts().sort_values(ascending=False)
dataset['VAR10'].mode()[0]
def impute_nan(df,variable):
    most_frequent_category=df[variable].mode()[0]
    df[variable].fillna(most_frequent_category,inplace=True)
impute_nan(dataset,'VAR10')

dataset.isnull().sum()

##VAR9 - Outliers are there - but need to reelook
dataset.boxplot("VAR9")
dataset.VAR9.hist(bins=50)
dataset = dataset.dropna(how='any', subset=['VAR9'])

dataset = dataset.drop(['VAR11'],axis=1)
dataset = dataset.drop(['VAR12'],axis=1)



def impute_nan(df,variable):
    dataset[variable+"_random"]=dataset[variable]
    
    random_sample=dataset[variable].dropna().sample(dataset[variable].isnull().sum(),random_state=0)
    
    random_sample.index=dataset[dataset[variable].isnull()].index
    dataset.loc[dataset[variable].isnull(),variable+'_random']=random_sample
impute_nan(dataset,"VAR13")
dataset = dataset.drop(["VAR13"],axis = 1)
dataset=dataset.rename(columns = {'VAR13_random':'VAR13'})



sns.distplot(dataset['VAR14'].dropna())
dataset.boxplot(column="VAR14")
dataset['VAR14'].describe()



uppper_boundary=dataset['VAR14'].mean() + 3* dataset['VAR14'].std()
lower_boundary=dataset['VAR14'].mean() - 3* dataset['VAR14'].std()
print(lower_boundary), print(uppper_boundary),print(dataset['VAR14'].mean())

dataset.loc[dataset['VAR14']>=71,'VAR14']=71
dataset.loc[dataset['VAR14']<=18,'VAR14']=18
dataset.VAR14.hist(bins=50)


dataset["VAR18"]=dataset.VAR18.fillna("Not Disclosed")
temp=dataset.groupby('VAR18')['VAR17'].mean()
find_set = dataset.VAR18.unique() 
for i in range(len(find_set)):
    a=str(find_set[i]).strip()
    #print(a)
    dataset.loc[(dataset['VAR18']== a) & (dataset['VAR17'].isna()==True),'VAR17']=temp[a]
dataset.boxplot(column="VAR17")

for i in range(len(find_set)):
    a=str(find_set[i]).strip()
    #print(a)
    dataset.loc[(dataset['VAR18']== a) & (dataset['VAR17']<5000),'VAR17']=temp[a]
dataset.boxplot(column="VAR17")


dataset = dataset.drop(["VAR27"],axis = 1)

lst_7=dataset.VAR10.value_counts().sort_values(ascending=False).head(7).index
lst_7=list(lst_7)
for categories in lst_7:
    dataset[categories]=np.where(dataset['VAR10']==categories,1,0)
A = dataset['VAR15'].value_counts()


lst_10=dataset.VAR15.value_counts().sort_values(ascending=False).head(10).index
lst_10=list(lst_10)
for categories in lst_10:
    dataset[categories]=np.where(dataset['VAR15']==categories,1,0)


dataset["VAR33"]=dataset.VAR33.fillna("Not Disclosed") 

dataset['VAR24'].mode()[0] 
def impute_nan(df,variable): 
    most_frequent_category=df[variable].mode()[0] 
    df[variable].fillna(most_frequent_category,inplace=True) 
impute_nan(dataset,'VAR24')

dataset['VAR5'].describe()
##Checking outliers
dataset.boxplot("VAR8")

import matplotlib.pyplot as plt
IQR=dataset.VAR8.quantile(0.75)-dataset.VAR8.quantile(0.25)
lower_bridge=dataset['VAR8'].quantile(0.25)-(IQR*1.5)
upper_bridge=dataset['VAR8'].quantile(0.75)+(IQR*1.5)
print(lower_bridge), print(upper_bridge)
dataset.loc[dataset['VAR8']>=31,'VAR8']=31
dataset.loc[dataset['VAR8']<=17,'VAR8']=17

dataset.isnull().sum()

dataset = dataset.drop(['VAR18'],axis=1)
dataset = dataset.drop(['VAR34'],axis=1)
dataset = dataset.drop(['VAR23','VAR37'],axis=1)
dataset = dataset.drop(['VAR28'],axis=1)
dataset = dataset.drop(['VAR10'],axis=1)
dataset = dataset.drop(['VAR15'],axis=1)
dataset = dataset.drop(['VAR35'],axis=1)
dataset.isnull().sum()


dataset = pd.get_dummies( dataset,drop_first = True )



X=dataset


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_scaled.columns

a=X.columns
X_scaled=pd.DataFrame(X_scaled,columns = a)

X_new = X_scaled[['VAR2', 'VAR5', 'VAR6', 'VAR9', 'VAR14', 'VAR31', 'VAR32', 'VAR36',
       'VAR8', 'HDFC BANK', 'EDM', 'Brokers & Small CA', 'Direct',
       'Other Banks & CA', 'B A', 'Graduation', 'H S C', 'S S C',
       'Under Matric (Class l to lX)', 'MBA', 'B E', 'B Tech',
       'VAR4_No Response', 'VAR7_Y', 'VAR16_Male', 'VAR21_Others',
       'VAR21_Salaried', 'VAR21_Self employed/ Business', 'VAR21_Student',
       'VAR22_Par', 'VAR24_DD', 'VAR24_ECS,SI',
       'VAR24_Online Credit/Debit Card/Teles Sales', 'VAR24_Online Netbanking',
       'VAR25_Savings', 'VAR26_Halfyearly Premium', 'VAR26_Monthly Premium',
       'VAR13_Tier III','VAR17']]


X_new.isnull().sum()


import pickle
with open('OOT_dataset.pkl', 'wb') as fv:
    pickle.dump(X_new, fv)


with open('x_res_obj.pkl', 'rb') as fv:
    X_res = pickle.load(fv)
with open('y_resobj.pkl', 'rb') as dfv:
    y_res = pickle.load(dfv)

with open('OOT_dataset.pkl', 'rb') as dfv1:
    OOT_data = pickle.load(dfv1)
    

X_res.value_counts()
from sklearn.ensemble import GradientBoostingClassifier 
#########Gradient Boosting
## Initializing Gradient Boosting with 500 estimators and max depth as 10.
gboost_clf = GradientBoostingClassifier( n_estimators=500, max_depth=10)
## Fitting gradient boosting model to training set
gboost_clf.fit(X_res, y_res )
predicted_classes_gboost = gboost_clf.predict(OOT_data)

predicted_test_data=pd.DataFrame(data=predicted_classes_gboost, columns=['predicted_value'])
predicted_test_data
predicted_test_data.to_csv("predicted_OOT_data_data.csv")
    