import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('Assignment_Train_Data.csv')
dataset.head()
dataset.columns

#dataset= dataset.rename(columns={'VAR1': 'Masked Policy Identifier', 'VAR2': 'Mapped Agent 13M Persistency', 'VAR3': 'Mapped Agent Branch', 'VAR4': 'Alcohol Declaration', 'VAR5': 'Annualized Premium', 'VAR6': 'Mapped Agent Vintage', 'VAR7': 'Auto Debit of Premium Opted'
#                                 ,'VAR8': 'BMI','VAR9': 'Risk Exposure of HDFC Life w.r.t. Life Assured','VAR10': 'Sourcing Channel','VAR11': 'Life Assured City','VAR12': 'Policy Contract Branch','VAR13': 'City Tier','VAR14': 'Age','VAR15': 'Education'
#                                 ,'VAR16': 'Gender','VAR17': 'Income','VAR18': 'Industry','VAR19': 'Marital Status'
#                                 ,'VAR20': 'Nationality','VAR22': 'Policy Tag','VAR23': 'Sourcing Partner','VAR24': 'Premium Payment Type','VAR25': 'Policy Product Category','VAR26': 'Payment Frequency','VAR27': 'Product Name','VAR28': 'Login Date','VAR29': 'Policy Price Sensitivity'
#                                 ,'VAR30': 'Residential Status','VAR31': 'Risk Cessation Term','VAR32': 'Policy Rider Opted Flag','VAR33': 'Smoker Declaration','VAR34': 'State','VAR35': 'Sourcing Sub Channel','VAR36': 'Policy Sum Assured','VAR37': 'HDFC Life Operational Zone','VAR38': 'Paid Premium Within 90 Days of Due Date','VAR21': 'Occupation'})
categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']
categorical_features
#dataset= dataset.rename(columns={'VAR21':'Occupation'})
dataset.isnull().sum()

#Preprocessing################################################

#Dropping the Policy Identifier
dataset = dataset.drop(['VAR1'],axis=1)
dataset.dtypes
dataset["VAR2"] = pd.to_numeric(dataset["VAR2"],errors = "coerce")

#distribution plot to see outliers
sns.distplot(dataset['VAR2'].dropna())
dataset.boxplot("VAR2")
dataset['VAR2'].describe()

dataset[dataset.VAR2.notnull()]

#Agent Persistency there were outliers as in the values ranges from 0 to 1 and then there some values which was greater than 1 and then values which are null in which there many , so considered null values as people who joined the departmetn so replaced na values with 0 and for the other values we standardised the values ranging more than 1 to 0's and 1's by dividing it by 100
for i in range(len(dataset)):
    if(dataset.VAR2[i]>1):
        dataset.VAR2[i]=dataset.VAR2[i]/100.0
#For Alcohol declaration we replaced the values which were null by creating another category with No response     
dataset['VAR2']=dataset.VAR2.fillna(0)
dataset = dataset.drop(["VAR3"],axis=1)
dataset['VAR4'] = dataset.VAR4.fillna('No Response')
#For Annualised Premium and Mapped agent vintage there were just 2 null values so we dropped those 2 rows so like this for columns which had all null values as 2 were dropped as in age,education,gender, marital status, nationality,Applicant's Policy PAR/NON PAR/ULIP Tag,Applicant's Policy Premium Payment Frequency,Application Life Assured Residential Status
dataset = dataset.dropna(how='any', subset=['VAR5', 'VAR6'])

#For BMI since there were around 34000 null values we thought imputation, 
#so for imputation we thought we will impute the null values by seeing the age and
#then then putting the ideal BMI for Age but that wont work so since the BMI were Missing values 
#at Random we randomly imputed the BMI values in the range of which is already present in the column say if values already present in the 
#dataset ranged from 18 to 30 this range was taken to impute .Also we found there were outliers in BMI after doinf this plotted the distribution graph and found 
#it was left skewed  so we FOUND THE upper and lower boundary using 1st Quartile - 3*IQR and .. we placed the upper values with upper boundary and the lower values with Lower boundary
def impute_nan(df,variable):
    dataset[variable+"_random"]=dataset[variable]
    
    random_sample=dataset[variable].dropna().sample(dataset[variable].isnull().sum(),random_state=0)
    
    random_sample.index=dataset[dataset[variable].isnull()].index
    dataset.loc[dataset[variable].isnull(),variable+'_random']=random_sample
impute_nan(dataset,"VAR8")
dataset = dataset.drop(["VAR8"],axis = 1)
dataset=dataset.rename(columns = {'VAR8_random':'VAR8'})


#Replaced null values of VAR 10 with highest frquency Category
dataset.VAR10.value_counts().sort_values(ascending=False)
dataset['VAR10'].mode()[0]
def impute_nan(df,variable):
    most_frequent_category=df[variable].mode()[0]
    df[variable].fillna(most_frequent_category,inplace=True)
impute_nan(dataset,'VAR10')

dataset.isnull().sum()

dataset.boxplot("VAR9")
dataset.VAR9.hist(bins=50)
dataset = dataset.dropna(how='any', subset=['VAR9'])

dataset = dataset.drop(['VAR11'],axis=1)
dataset = dataset.drop(['VAR12'],axis=1)

#For City Tier since it is an Ordinal Variable and it missing values at random(3127) so we randomly imputed the values

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


#For age we plotted the outliers and found there were age 
#which were negative and some values we plotted the distribution plot and 
#it was normal distributed so we got the upper boundary and lower boundary (using mean +3sd and mean -3sd) so for we replaced Values which were less than 18 as 18 and for the upper boundary we replaced the values above 71 as 71 (since 71 was the upper boundary)
uppper_boundary=dataset['VAR14'].mean() + 3* dataset['VAR14'].std()
lower_boundary=dataset['VAR14'].mean() - 3* dataset['VAR14'].std()
print(lower_boundary), print(uppper_boundary),print(dataset['VAR14'].mean())

dataset.loc[dataset['VAR14']>=71,'VAR14']=71
dataset.loc[dataset['VAR14']<=18,'VAR14']=18
dataset.VAR14.hist(bins=50)


dataset["VAR18"]=dataset.VAR18.fillna("Not Disclosed")

#Income - replaced NA Values with mean income of industry column & for Income less than 5000 -replaced with 5000
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

#Education -Also after this we found there too many categories so which creating dummy variables it will increase the dimensions so for treating this we replaced the top 10 Frequent categories as 1's and the other as 0's
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



dataset = pd.get_dummies( dataset,drop_first = True )


#FEATURE SELECTION##################################################3
X=dataset.drop(['VAR38'],axis=1)
Y=dataset['VAR38']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X_scaled,Y)
print(model.feature_importances_)
    
ranked_features=pd.Series(model.feature_importances_,index=X.columns)
ranked_features.nlargest(20).plot(kind='barh')
plt.show()

#Auto Debit of Premium Opted Flag is the most important feature in the dataset

#check correlation
#Checking VIF




from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_scaled,Y)
selected_feat = X.columns[(feature_sel_model.get_support())]

print('total features: {}'.format((X.shape[1])))
print('selected features: {}'.format(len(selected_feat)))

selected_feat

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

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]
vif['variable'] = X_new.columns

Y.value_counts()

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss

# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state=42)
X_res,y_res=smk.fit_sample(X_new,Y)

X_res.shape,y_res.shape

import pickle
with open('x_res_obj.pkl', 'wb') as fv:
    pickle.dump(X_res, fv)
with open('y_resobj.pkl', 'wb') as dfv:
    pickle.dump(y_res, dfv)
#Saving as an Object to be used in the Modeling file
    
############################################################################