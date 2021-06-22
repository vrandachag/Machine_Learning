import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("HRDataset_v14.csv")

print(df)

d = {'US citizen':1,'Eligible NonCitizen':-1,'Non-Citizen':0}
df['CitizenDesc'] = df['CitizenDesc'].map(d)

d = {'Y': 1,'N':0}
df['Project'] = df['Project'].map(d)

d = {'Active': 1, 'Voluntarily Terminated': 0, 'Terminated for Cause':0}
df['EmploymentStatus']=df['EmploymentStatus'].map(d)

features = ['DeptID','Salary','CitizenDesc','EmploymentStatus']
X = df[features];
y = df['Project']

print(X)
print(y);
