import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('CPNG.csv')
df['Date'] = pd.to_datetime(df.Date)
df.index = df['Date']
print(df.index)
df.index = (df.index - pd.to_datetime('2021-01-01')).days

print("df index print")
print(df.index)
print(type(df.index))

#convert input to array
y = np.asarray(df['Close'])
x = np.asarray(df.index.values)

regression_model = LinearRegression()
regression_model.fit(x.reshape(-1,1),y.reshape(-1,1))

y_learned = regression_model.predict(x.reshape(-1,1));

newindex = np.asarray(pd.RangeIndex(start=x[-1],stop=x[-1] + 200))

y_predict = regression_model.predict(newindex.reshape(-1,1))
               
print("Closing price at Jan 2022 would be around ", y_predict[-1])

x = pd.to_datetime(df.index, origin = '2021-01-01',unit='D')
future_x = pd.to_datetime(newindex, origin='2021-01-01', unit='D')

#rcParams['figure.figsize']=20,10

plt.figure(figsize=(16,8))
plt.plot(x,df['Close'],label = 'Close Price History')

plt.plot(x,y_learned,color = 'r', label = 'Mathematical Model')

plt.plot(future_x,y_predict,color='g',label='Furture predictions')

plt.suptitle('Stock Market Predictions', fontsize = 16)

fig = plt.gcf()
fig.canvas.set_window_title("Stock Market Predictions")

plt.legend()
plt.show()
