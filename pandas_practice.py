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

print("DESC:")
print(df)
d = {'Y': 1,'N':0}
df['Project'] = df['Project'].map(d)

d = {'Active': 1, 'Voluntarily Terminated': 0, 'Terminated for Cause':0}
df['EmploymentStatus']=df['EmploymentStatus'].map(d)

features = ['Salary','CitizenDesc','EmploymentStatus']
X = df[features]
y = df['Project']

print(X)
print(y)

dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X,y)
#tree.plot_tree(dtree)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = sklearn.tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
