import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 

df=pd.read_csv("Car.csv")


#Cleaning
print(df.isna().sum())

df=df.dropna()
df=df.reset_index(drop=True)

'''
Analyse each and every column with respect to dependent
'''

'''
Seats
'''
print(df['seats'].value_counts())#show seats with count

'''
torque
'''
print(df['torque'].dtype)

df=df.drop(columns='torque')
#Because of unstructure pattern we have to drop the column

'''
max power
'''
print(df['max_power'].dtype)
df['max_power']=df['max_power'].apply(lambda x:x.split()[0])


#df['max_power']=df['max_power'].astype("float32")
#Above line five error find the noise value
l=[]
for i in range(len(df)):
    try:
        float(df.iloc[i,-2])
    except:
        l.append(i)

df=df.drop(index=l).reset_index(drop=True)
df['max_power']=df['max_power'].astype("float32")   

'''
Engine
'''
df['engine']=df['engine'].apply(lambda x:x.split()[0])
#df['engine']=df['engine'].astype('float32')

l=[]
for i in range(len(df)):
    try:
        float(df.iloc[i,-3])
    except:
        l.append(i)

df=df.drop(index=l).reset_index(drop=True)
df['engine']=df['engine'].astype("float32") 

'''
Mileage
'''

df['mileage']=df['mileage'].apply(lambda x:x.split()[0]).astype('float32')


'''
Owner
'''
print(df['owner'].value_counts())

df['owner']=df['owner'].replace({"Fifth":"Fourth & Above Owner"})

print(df['owner'].value_counts())

#Here we remove test drive car
f=df['owner']=='Test Drive Car'
df=df.drop(index=df[f].index).reset_index(drop=True)
print(df['owner'].value_counts())
#Reason of dropping:-Test drive car were considered as outlier in our data 
#thats why we drop it


#Make a violin plot of each category in seats column with respect to selling price 
#using loop, Make the conclusion of that column wrt selling price.
# data=df.groupby('seats')['selling_price'].mean()
# seats=df['seats'].unique()
# plt.violinplot(data.index,data)
# plt.xlabel("Seats")
# plt.ylabel("Selling Price")
# plt.show()

'''
Transmission
'''

print(df['transmission'].value_counts())

'''
seller type
'''

print(df['seller_type'].value_counts())

for x in df['seller_type'].unique():
    f=df['seller_type']==x
    plt.violinplot(df.loc[f,'selling_price'])
    plt.title(x)
    plt.ticklabel_format(style='plain')
    plt.show()
    
'''
Fuel
'''

print(df['fuel'].value_counts())    
for x in df['fuel'].unique():
    f=df['fuel']==x
    plt.violinplot(df.loc[f,'selling_price'])
    plt.title(x)
    plt.ticklabel_format(style='plain')
    plt.show()
    
#Conclusio-->There is similar distribution of CNG and LPG 
#Petrol and Diesel so merge these categories
df['fuel']=df['fuel'].replace({'CNG':0,'LPG':0,'Petrol':1,'Diesel':1})


'''
Name
'''

print(df['name'].value_counts())

df['name']=df['name'].apply(lambda x:x.split()[0])
#The main impact is of brand so just turn car name column into brand only

brands=df['name'].unique()

#Here name is non-ordinal but one hot encoding is not possible because after 
#doing one hot encoding 30+ columns made

#When number of categories is high then make groups of categories according
#to dependent variable

brand_selling_price=df.groupby('name')['selling_price'].mean()

brand_selling_price=brand_selling_price.sort_values(ascending=False)

def fxn(x):
    if x in brand_selling_price.iloc[:10]:
        return 0
    elif x in brand_selling_price.iloc[10:25]:
        return 1
    else:
        return 2
    
    
df['name']=df['name'].apply(fxn)


'''
Statistics analysis of columns
'''

categorical=df[['name','fuel','seller_type','transmission','owner','seats']]
numeric=df[['year','km_driven','mileage','engine','max_power','selling_price']]

from sklearn.preprocessing import LabelEncoder
encoder1=LabelEncoder()
categorical['owner']=encoder1.fit_transform(categorical['owner'])


from sklearn.preprocessing import LabelEncoder
encoder2=LabelEncoder()
categorical['transmission']=encoder2.fit_transform(categorical['transmission'])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
onehotencode=OneHotEncoder(drop='first')
ct=ColumnTransformer([('encode',onehotencode,[2])],remainder='passthrough')
categorical=ct.fit_transform(categorical)

categorical=pd.DataFrame(categorical)

'''
Feature Selection
'''

#Four ways of feature selection

#1-->Independent variable is numeric and dependent is also numeric
#2-->Independent variable is numeric and dependent is category
#3-->Independent variable is category and dependent is numeric
#4-->Independent variable is category and dependent is category



#Methods for feature selection------------>
#If one column is numeric and other is categorical--->ANOVA(Analysis of Variance)
#If one column is numeric and other also numeric---->Pearson correlation method
#If both columns are categorical---->Chi Square Test

'''
Pearson correlation method
'''
#For checking correlation between two numeric  column use pearson correlation 
#method
#In this method we find the correlation coefficient(r)

corr=numeric.corr()   #It returns a correlation matrix

#To represent the correlation matrix in graphic way use heatmap
import seaborn
seaborn.heatmap(corr)
plt.show()

#Heatmap is used to represent the correlation between two columns

#if in corr--> max_power and engine value is 0.99,0.98 or 1 anything than
#what is the conclusion in this
#Note:-If two independent columns have strong correlation between tham, that
#means both columns are impacting the dependent columns in same way. So we can
#drop one column from it..


'''
ANOVA  (Analysis of Variance)
'''

#Anova is used to find out wether categories have same variance with respect 
#to numeric(dependent) column or not

#In hypotheses testing we have two assumptions(cases)
#H0-->(Null hypothesis)-->All categoriews have same variance
#H1-->(Alternavtive hypothesis)-->Variance of all categories is different


#After testing we have two either reject or accept null hypothesis


#Steps to perform ANOVA--------------->
#Make assumptions (H0 AND H1)
#Calculate within sum of square and between sum of square
#Calculate degree of freedom of groups and each category 
#Find f ->value
#Take alpha value and confidence interval and find in graph wether to accept 
#or reject null hypothesis



#Within sum of square is square difference between each sample and thier group
#mean-->(SSW)

#Betweeen sum of square is square difference between each category's sample and 
#grand total mean of all categories-->(SSB)



#F value represent if the test is statical significant 
#DOF(B)-->Degree of freedom in group
#DOF(W)-->Degree of freedom between samples
#F=(SSB/DOF(B)/(SSW/DOF(W))

#Degree of freedom is number of samples that can freely move in data without 
#breaking the constraint



from sklearn.feature_selection import SelectKBest,f_classif
#Select K best is used to sort the best columns according to test value passed
#as parameter
sk=SelectKBest(f_classif,k=7)
#k value represent the number of columns we want
#This value depends upon our domain knowledge

result=sk.fit_transform(categorical,numeric['selling_price'])

#In result there are top n columns with highest f score 
print(sk.scores_)




'''
Outlier Detection
'''
# Outliers are those points which does not lie under the general pattern of a
# column
#POutliers are always checked in numeric

plt.scatter(numeric['km_driven'],numeric['selling_price'])
plt.show()


#Two cases:---
#When column is normally distributed
#When column does not have any distributuion

#Plot Graph

for x in numeric.columns:
    plt.hist(numeric[x])
    plt.title(x)
    plt.show()
    
#To detect the outliers in gaussion distribution
#Q-Outlier
#Z-score

#To detect the outliers in non gaussion distribution:
#DBSCAN-->Density based spatial clustring of application with noise

#Z score is used to check the deviation of a point from the mean

def z_score(column):
   mean=column.mean()
   std=column.std()
   z=np.abs((column-mean)/std)
   return column[z>3]



outliers1=z_score(numeric['km_driven'])

outliers2=z_score(numeric['max_power'])

outliers3=z_score(numeric['mileage']) 

print(len(outliers1)+len(outliers2)+len(outliers3))   

#Task-->Drop the rows of outliers

f=~numeric['km_driven'].isin(z_score(numeric['km_driven']))
numeric=numeric[f]
categorical=categorical[f]
f=~numeric['max_power'].isin(z_score(numeric['max_power']))
numeric=numeric[f]
categorical=categorical[f]
f=~numeric['mileage'].isin(z_score(numeric['mileage']))
numeric=numeric[f]
categorical=categorical[f]

# numeric=numeric[~numeric['km_driven'].isin(z_score(numeric['km_driven']))]
# numeric=numeric[~numeric['max_power'].isin(z_score(numeric['max_power']))]
# numeric=numeric[~numeric['mileage'].isin(z_score(numeric['mileage']))]






'''
DBSCAN
'''
#Outlier detection, Fraud detection,Clustering
#DBSCAN-->Density based spatial clustring of application with noise



engine=numeric[['engine','selling_price']]

#To know the efficient value of epsilon draw k-distance graph
#K-distance graph tells us the overall average distance of nearest neighbor
#of each point

#Steps to Make K-Distance Graph------------------>

#Step1-->To deal with distance always scale down your data first

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
engine=scaler.fit_transform(engine)


from sklearn.neighbors import NearestNeighbors
#NearestNeighbors is a class use to find the number of neighbors and thier 
#distances in array

neighbor=NearestNeighbors(n_neighbors=10)
#It will find 10 nearest neighbors along with thier distances
#Nearest neighbors return means its index

#Step2-->Finding nearest neighbors
neighbor.fit(engine)

#Step3-->Finding the distance
distance,index=neighbor.kneighbors(engine)
#Each row of this array represent the distance of that point with its n neighbors


#Step4-->Sort the distance
distance=np.sort(distance,axis=0)

#Step5-->Extract out distance of nearest neigbors
distance=distance[:,1]

#Step6-->Make distance graph
plt.plot(distance)
plt.title("K-Distance Graph")
plt.show()
#For best epsilon choose value where graph start drastically increase

#Applying DBSCAN algorithm
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.05,min_samples=10)
model=dbscan.fit(engine)

points=model.labels_

plt.scatter(numeric['engine'],numeric['selling_price'],c=points)
plt.show()

#Below filtering will keep only those points whose respective labels(noise) is
#not equal to -1 
numeric=numeric[points!=-1]

categorical=categorical[points!=-1]

#Combine numeric and categorical
#STD Scaling
#Build regression model and find metrics

#SVM,linear regresson,Decision tree --->find best r2_score

df1=pd.concat((categorical,numeric),axis=1,ignore_index=True)


X=df1.iloc[:,:-1]
y=df1.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)




'''
Random Forest Regressor
'''

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
print(y_pred)


from sklearn.metrics import mean_absolute_error as mae,r2_score
print(mae(y_test,y_pred))
print('r2_score',r2_score(y_test,y_pred))



'''
Taking user input
'''


#Task-->Get data from input and predict its value
#Hint-->First change the live data into format used in training (Don't use 
#feature selection and outliers )
# one hot encoding transform
# label transform
# scaling
# data=sc.transform(data)


import pandas as pd
import numpy as np


data={'name':[],'year':[],'km_driven':[],'fuel':[],'seller_type':[],'transmission':[],
      'owner':[],'mileage':[],'engine':[],'max_power':[],'seats':[]}

n=input("Enter the name of the car:")
data['name'].append(n)
y=input('Enter the year of purchase:')
data['year'].append(y)
km=input("Enter the km driven:")
data['km_driven'].append(km)
f=input('Enter the fuel type:')
data['fuel'].append(f)
s=input("Enter the type of seller(Individual/Dealer):")
data['seller_type'].append(s)
t=input("Enter the transmission:")
data['transmission'].append(t)
o=input("Enter the owner(First Owner/Second Owner/Third):")
data['owner'].append(o)
m=input("Enter the mileage (kmpl):")
data['mileage'].append(m)
e=input("Enter the engine (cc):")
data['engine'].append(e)
mp=input("Enter the max power (bhp):")
data['max_power'].append(mp)
se=input("Enter the seats:")
data['seats'].append(se)

data=pd.DataFrame(data) 


'''
Analyse each and every column with respect to dependent
'''

print(data['seats'].value_counts())#show seats with count

'''
max power
'''
print(data['max_power'].dtype)

data['max_power']=data['max_power'].apply(lambda x:x.split()[0])

#df['max_power']=df['max_power'].astype("float32")
#Above line five error find the noise value
l=[]
for i in range(len(data)):
    try:
        float(data.iloc[i,-2])
    except:
        l.append(i)

data=data.drop(index=l).reset_index(drop=True)
data['max_power']=data['max_power'].astype("float32")   

'''
Engine
'''
data['engine']=data['engine'].apply(lambda x:x.split()[0])
#df['engine']=df['engine'].astype('float32')

l=[]
for i in range(len(data)):
    try:
        float(data.iloc[i,-3])
    except:
        l.append(i)

data=data.drop(index=l).reset_index(drop=True)
data['engine']=data['engine'].astype("float32") 

'''
Mileage
'''

data['mileage']=data['mileage'].apply(lambda x:x.split()[0]).astype('float32')


'''
Owner
'''
print(data['owner'].value_counts())

data['owner']=data['owner'].replace({"Fifth":"Fourth & Above Owner"})




'''
Transmission
'''

print(data['transmission'].value_counts())

'''
seller type
'''

print(data['seller_type'].value_counts())

    
'''
Fuel
'''


    
#Conclusio-->There is similar distribution of CNG and LPG 
#Petrol and Diesel so merge these categories
data['fuel']=data['fuel'].replace({'CNG':0,'LPG':0,'Petrol':1,'Diesel':1})


'''
Name
'''

print(data['name'].value_counts())

data['name']=data['name'].apply(lambda x:x.split()[0])



data['name']=data['name'].apply(fxn)






'''
Statistics analysis of columns
'''

categorical=df[['name','fuel','seller_type','transmission','owner','seats']]
numeric=df[['year','km_driven','mileage','engine','max_power']]


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
categorical['owner']=encoder.fit_transform(categorical['owner'])


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
categorical['transmission']=encoder.fit_transform(categorical['transmission'])


categorical=ct.transform(categorical)

categorical=pd.DataFrame(categorical)

#sparse=False will not allow to form the sparse matrix after one hot encode 


#Confine numeric and categorical
#Scaling ->Standard scling
#Build Regression model and find metrics

df1=pd.concat((categorical,numeric),axis=1,ignore_index=True)



'''
Scaling
'''
bb=df1
bb=sc.transform(bb)


'''
random forest regressor
'''
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
car_price=y_pred[-1]


print("Car Price : ",car_price)
