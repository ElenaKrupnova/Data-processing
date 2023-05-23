# создание классов "Товар" и "Список товаров" с возможностью добавлять или убирать товары, а также подсчитывать общую
# стоимость покупки

class Good:
    def __init__(self, name, volume, price, country):
        self.name = name
        self.volume = volume
        self.price = price
        self.country = country

class GoodList:

    def __init__(self, owner, date):
        self.l = list()
        self.owner = owner
        self.date = date

    def add(self, *args: Good):
        for arg in args:
            self.l.append(arg)

    def __str__(self):
        g_str = ""
        for g in self.l:
            g_str += g.name + " "
        return f"{self.owner}. Your goods: {g_str}"

    def remove_one(self, g_name):
        for i in range(len(self.l)):
            if self.l[i].name == g_name:
                self.l.pop(i)
                break

    def remove_all(self, g_name):
        for i in range(len(self.l)-1,-1,-1):
            if self.l[i].name == g_name:
                self.l.pop(i)

    def total(self):
        count = 0
        for g in self.l:
            count += g.price
        return count

shower_gel = Good("shower gel Delicare dessert donuts", 400, 249, 'Russia')
brow_pencil = Good("brow pencil Eveline", 4, 199, 'Poland')
face_mask = Good("face mask El'Skin", 23, 287, 'South Korea')

g1 = GoodList("Maria Ivanchenko", 05.2023)
g1.add(shower_gel, face_mask, face_mask)
g1.add(Good("kids shampoo", 350, 189, 'Russia'))
print(g1)
print(g1.total())
g1.remove_all("face mask El'Skin")
print(g1)
print(g1.total())

import pandas as pd
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import style
from sklearn.model_selection import train_test_split

# чтение файла с составленным датасетом, подготовка данных к прогнозированию

df = pd.read_csv("dataset.csv", on_bad_lines='skip')
df = df.drop(['Good', 'Volume', 'Country'], axis=1)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df1 = df.groupby(['Date']).agg(Price_sum=('Price', 'sum'))

# прогнозирование новых данных с использованием модели линейной регрессии
X = df1.iloc[:, 0]
y = df1.iloc[:, -1]

plt.scatter(X, y)
plt.xlabel('Сумма покупки')
plt.ylabel('Дата покупки')
plt.show()

model = LinearRegression()
scaler = MinMaxScaler()

X = scaler.fit_transform(pd.DataFrame(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)

# предсказание на несколько лет вперёд

plt.figure(figsize=(16,7))
plt.plot(df1)
plt.title('Стоимость покупок с 2018 по 2020 годы')
plt.xlabel('Дата покупки')
plt.ylabel('Сумма покупок')
plt.show()

style.use('ggplot')
df1.dropna(inplace=True)

last_close = df1['Price_sum'][-1]
last_date = df1.iloc[-1].name.timestamp()
df1['Forecast'] = np.nan

for i in range(1000):
    modifier = random.randint(-100, 105) / 10000 + 1
    last_close *= modifier
    next_date = datetime.datetime.fromtimestamp(last_date)
    last_date += 86400

    df1.loc[next_date] = [np.nan, last_close]

df1['Price_sum'].plot()
df1['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Дата покупки')
plt.ylabel('Стоимость покупки')
plt.show()