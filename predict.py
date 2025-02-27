import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegression:
    def __init__(self, x, y):
        self.data = x
        self.label = y
        self.m = 0  # Slope
        self.b = 0  # Intercept
        self.n = len(x)

    def predict(self, input):
        y_pred = self.m * input + self.b
        return y_pred


df = pd.read_csv('data.csv')
x = np.array(df.iloc[:, 0])
y = np.array(df.iloc[:, 1])

model = LinearRegression(x, y)

y_pred = model.predict(10)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue')
plt.plot(x, model.m * x + model.b, color='red')
plt.title('Linear Regression')
plt.xlabel('x', size=20)
plt.ylabel('y', size=20)
plt.show()
