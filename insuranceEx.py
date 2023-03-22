import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  # MOdeli olsuturuyruz
from keras.layers import Dense  # katrmanları olusturuyoruz
from sklearn.metrics import mean_squared_error, mean_absolute_error


dataFrame = pd.read_csv("insurance.csv")

print(dataFrame)
print(dataFrame.head())
print(dataFrame.describe())
print(dataFrame.isnull().sum())

newGender = []
for data in dataFrame["sex"]:
    if data == "male":
        data = 0
        newGender.append(data)
    else:
        data = 1
        newGender.append(data)
# print(newData)
dataFrame["sex"] = newGender
# print(dataFrame["sex"])
print(dataFrame.describe())

sbn.displot(dataFrame["region"])
plt.show()

newSmoke = []
for data in dataFrame["smoker"]:
    if data == "yes":
        data = 0
        newSmoke.append(data)
    else:
        data = 1
        newSmoke.append(data)

dataFrame["smoker"] = newSmoke
print(dataFrame.describe())

print(dataFrame.corr()["smoker"].sort_values)

sbn.displot(dataFrame["charges"])
plt.show()
print(dataFrame.sort_values("charges", ascending=False))

yeniDataFrame = dataFrame.sort_values("charges", ascending=False).iloc[7:]

sbn.displot(yeniDataFrame["charges"])
plt.show()
dataFrame = yeniDataFrame

dataFrame = dataFrame.drop("region", axis=1)

y = dataFrame["smoker"].values
x = dataFrame.drop("smoker", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=10)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)

model = Sequential()

model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))

model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.fit(x=x_train, y=y_train, validation_data=(
    x_test, y_test), batch_size=100, epochs=300)

kayipVerisi = pd.DataFrame(model.history.history)
# Burdaki veride birbirinden ayrı cıksaydı epochu yada baska bir seyi degistirmek zorundaydık
print(kayipVerisi)
kayipVerisi.plot()
plt.show()


tahminDizisi = model.predict(x_test)

print(tahminDizisi)

mean_absolute_error(y_test, tahminDizisi)

plt.scatter(y_test, tahminDizisi)
plt.plot(y_test, y_test, "g*-")
plt.show()
