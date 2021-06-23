import pandas
import graphviz
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("HRDataset_v14.csv")

print(df)

d = {'US Citizen':1,'Eligible NonCitizen':-1,'Non-Citizen':0}
df['CitizenDesc'] = df['CitizenDesc'].map(d)

d = {'Y': 1,'N':0}
df['Project'] = df['Project'].map(d)

d = {'Active': 1, 'Voluntarily Terminated': 0, 'Terminated for Cause':0}
df['EmploymentStatus']=df['EmploymentStatus'].map(d)

features = ['Salary','CitizenDesc','EmploymentStatus']
X = df[features]
y = df['Project']

print(X)
print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)


#worked
tree.plot_tree(dtree,filled = True,rounded=True,precision=2,fontsize=12)
plt.savefig("decisiontree.png")

print(dtree.predict([[25000,-1,1]]))

plt.show()
