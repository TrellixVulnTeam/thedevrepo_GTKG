import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('Hackerrank/LaptopBatteryLifePrediction/trainingdata.txt',header=None, names=['battery_charging_time', 'battery_lasted_time'])
print("Head: {}".format(data.head()))
print("Correlation: {}".format(data.corr()))

data.plot('battery_charging_time', 'battery_lasted_time', kind='scatter')
plt.show()

print("X and y are highly correlated (0.82) => there's linearity.\
\nHowever, y gets saturated at 8 as seen from the plot. Therefore, the relationship isn't linear beyond a point.")


X,y = data.battery_charging_time, data.battery_lasted_time

X = X.values.reshape(-1,1)
y = y.values.reshape(-1,1)

print(X.shape, y.shape)

clf = LinearRegression(normalize=False)
clf.fit(X,y)

print("Coeffecients: {}\t\t Intercept: {}".format(clf.coef_, clf.intercept_))

input = float(input())
print("Prediction for {}: {}".format(input, clf.predict([[input]])))

# Simple tweak
output = round(2*input,2)
if output > 8.0:
    output = 8.0

print("Simple tweak for prediction of {}: {}".format(input, output))
