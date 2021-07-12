import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

df = pd.read_csv('Book1.csv')

df['Date'] = pd.to_datetime(df.Date)
df.index = df['Date']

df.index = (df.index - pd.to_datetime('2021-03-21')).days

df['Open-Close'] = (df.Open - df.Close)/df.Open
df['High-Low'] = (df.High - df.Low)/df.Low
df['percent_change'] = df['Adj Close'].pct_change()
df['std_5'] = df['percent_change'].rolling(5).std()
df['ret_5'] = df['percent_change'].rolling(5).mean()
df.dropna(inplace=True)

print(df['High-Low'])

#input variable
X = df[['Open-Close','High-Low','std_5','ret_5']]

#target or output variable
y = np.where(df['Adj Close'].shift(-1) > df['Adj Close'], 1 , -1)

# Total dataset length
dataset_length = df.shape[0]

# Training dataset length
split = int(dataset_length * 0.75)
split

# Splitiing the X and y into train and test datasets
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Print the size of the train and test dataset
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

clf = RandomForestClassifier(random_state=5)
model=clf.fit(X_train, y_train)

print('Correct Prediction (%): ',accuracy_score(y_test, model.predict(X_test), normalize = True)*100.0)

report = classification_report(y_test,model.predict(X_test))
print(report)

df['strategy_returns'] = df.percent_change.shift(-1)*model.predict(X)


#matplotlib inline
df.strategy_returns[split:].hist()
plt.xlabel('Strategy returns (%)')
plt.show()
'''
(df.strategy_returns[split:]+1).cumprod().plot()
plt.ylabel('Strategy returns (%)')
plt.show()
'''
