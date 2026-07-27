import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

y = np.array([2.1, 4.2, 5.8, 8.1, 9.9, 12.2, 14.1, 15.8, 18.2, 20.1])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)


print("Hệ số β:", model.coef_)
print("Bias β0:", model.intercept_)


y_pred = model.predict(X_test)

print("\nGiá trị thực tế:")
print(y_test)

print("\nGiá trị dự đoán:")
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMSE =", mse)
print("R2 Score =", r2)

x_new = np.array([[11]])

prediction = model.predict(x_new)

print("\nDự đoán khi X = 11:", prediction[0])

plt.scatter(X, y, label="Dữ liệu thực tế")

plt.plot(
    X,
    model.predict(X),
    label="Ridge Regression"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Ridge Regression")

plt.legend()
plt.show()