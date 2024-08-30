import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv(r"C:\Users\nk452\Supportvector\svr\Position_Salaries.csv")

x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

y=y.reshape(len(y),1)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)
print(x)
print(y)
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x,y)
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='red')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1).reshape(-1,1)),color='blue')
plt.title('graph of (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
