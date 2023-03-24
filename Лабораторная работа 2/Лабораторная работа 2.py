# 1) Есть два файла с данными турагенства: email.csv и username.csv. C ними нужно проделать все манипуляции, указанные
# в лекции 2, а именно: a) Группировка и агрегирование (сгруппировать набор данных по значениям в столбце, а затем
# вычислить среднее значение для каждой группы) b) Обработка отсутствующих данных (заполнение отсутствующих значений
# определенным значением или интерполяция отсутствующих значений) c) Слияние и объединение данных (соединить два
# DataFrames в определенном столбце)
import pandas as pd

df = pd.read_csv('email.csv')
df2 = pd.read_csv('username.csv')
grouped_data_1 = df.groupby('Total price').mean()
grouped_data_2 = df2.groupby('Trip count').mean()
print(grouped_data_1)
print(grouped_data_2)

df = df.fillna(value=0)
df2 = df2.fillna(value=0)
print(df, df2)

merged_data = pd.merge(df, df2, on='Identifier')
print(merged_data)

# 2. a) Нужно создать сводную таблицу так, чтобы в index были столбцы “Rep”, “Manager” и “Product”, а в values “Price” и
# “Quantity”. Также нужно использовать функцию aggfunc=[numpy.sum] и заполнить отсутствующие значения нулями. В итоге
# можно будет увидеть количество проданных продуктов и их стоимость, отсортированные по имени менеджеров и директоров.
# b) Учебный файл (data.csv) + практика Dataframe.pivot. Поворот фрейма данных и суммирование повторяющихся значений.
import numpy as np
import pandas as pd

df = pd.read_csv('sales.csv')
df = df.fillna(value=0)
index = ['Rep', 'Manager', 'Product']
columns = ['Account', 'Name', 'Status']
values = ['Price', 'Quantity']
pivoted_data = df.pivot_table(index=index, columns=columns, values=values, aggfunc=[np.sum])
print(pivoted_data)

import pandas as pd
df = pd.read_csv('data.csv')
pivoted_data = df.pivot_table(index='Date', columns='Product', values='Sales',aggfunc='sum')
print(pivoted_data)

# 3) Визуализация данных (можно использовать любой из учебных csv-файлов).
# a) Необходимо создать простой линейный график из файла csv (два любых столбца, в которых есть зависимость)
# b) Создание визуализации распределения набора данных. Создать произвольный датасет через np.random.normal или
# использовать датасет из csv-файлов, потом построить гистограмму.
# c) Сравнение нескольких наборов данных на одном графике. Создать два набора данных с нормальным распределением или
# использовать данные из файлов. Оба датасета отразить на одной оси, добавить легенду и название.
# d) Построение математической функции. Создать данные для x и y (где x это numpy.linspace, а y - заданная в условии
# варианта математическая функция). Добавить легенду и название графика. Вариант 1 - функция sin  Вариант 2 - функция
# cos.
# e) Моделирование простой анимации. Создать данные для x и y (где x это numpy.linspace, а y - математическая функция).
# Запустить объект line, ввести функцию animate(i) c методом line.set_ydata() и создать анимированный объект
# FuncAnimation. a)	Шаг 1: смоделировать график sin(x) (или cos(x)) в движении. b)	Шаг 2: добавить к нему график
# cos(x) (или sin(x)) так, чтобы движение шло одновременно и оба графика отображались на одной оси.
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np

df = pd.read_csv('cars.csv')
plt.scatter(df['Car'], df['Horsepower'])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple line of df')
plt.show()

data = np.random.normal(30, 20, 80)
plt.hist(data, bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()

data1 = np.random.normal(40, 10, 100)
data2 = np.random.normal(70, 25, 100)
fig, ax = plt.subplots()
ax.plot(data1, label='Dataset 1')
ax.plot(data2, label='Dataset 2')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('Comparison of Two Datasets')
ax.legend()
plt.show()

x = np.linspace(-np.pi, np.pi, 80)
y = np.cos(x)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Plot of the Cos Function')
plt.show()

x = np.linspace(0, 2 * np.pi, 100)
y = np.cos(x)
y1 = np.sin(x)
fig, ax = plt.subplots()
line, = ax.plot(x, y)
line1, = ax.plot(x, y1)
def animate(i):
    line.set_ydata(np.cos(x + i / 10))
    return line,
def animate_2(i):
    line1.set_ydata(np.sin(x + i / 10))
    return line1,
ani = animation.FuncAnimation(fig, animate, frames=100, blit=False)
ani1 = animation.FuncAnimation(fig, animate_2, frames=100, blit=False)
plt.show()

# 4) Загрузка CSV-файла в DataFrame. Используя pandas, напишите скрипт, который загружает CSV-файл в DataFrame и
# отображает первые 5 строк df.
import pandas as pd

df = pd.read_csv('cars.csv')
print(df.head())

# 5) Выбор столбцов из DataFrame. a) Используя pandas, напишите сценарий, который из DataFrame файла sales.csv
# выбирает только те строки, в которых Status = presented, и сортирует их по цене от меньшего к большему. b) Из файла
# climate.csv отображает в виде двух столбцов названия и коды (rw_country_code) тех стран, у которых cri_score больше
# 100, а fatalities_total не более 10. c) Из файла cars.csv отображает названия 50 первых американских машин, у
# которых расход топлива MPG не менее 25, а частное внутреннего объема (Displacement) и количества цилиндров не более
# 40. Названия машин нужно вывести в алфавитном порядке.
import pandas as pd

df_1 = pd.read_csv('sales.csv')
condition = df_1['Status'] == 'presented'
new_df = df_1[condition]
print(new_df.sort_values('Price', ascending=True))

df_2 = pd.read_csv('climate.csv')
condition_1 = (df_2['cri_score'] > 100) & (df_2['fatalities_total'] < 10)
new_df_1 = df_2[condition_1]
columns = ['rw_country_code', 'rw_country_name']
print(new_df_1[columns])

df_3 = pd.read_csv('cars.csv')
condition_2 = (df_3['Origin'] == 'US') & (df_3['MPG'] > 25) & ((df_3['Displacement'] / df_3['Cylinders']) < 40)
new_df_2 = df_3[condition_2]
print(new_df_2.head(50).sort_values('Car', ascending=True))

# 6) Используя numpy, напишите скрипт, который загружает файл CSV в массив numpy и вычисляет среднее значение,
# стандартное отклонение и максимальное значение массива. Для тренировки используйте файл data.csv, а потом любой
# другой csv-файл от 20 строк.
import numpy as np

df_data = np.genfromtxt('data.csv', delimiter=',', usecols=(1,2), skip_header=1)
mean = np.mean(df_data)
std = np.std(df_data)
max_value = np.max(df_data)
print("Mean:", mean)
print("Standard deviation:", std)
print("Max value:", max_value)

data = np.genfromtxt('tomato.csv', delimiter=',', usecols=(2,3,4), skip_header=1)
mean = np.mean(data)
std = np.std(data)
max_value = np.max(data)
print("Mean:", mean)
print("Standard deviation:", std)
print("Max value:", max_value)

# 7) Используя numpy, напишите сценарий, который создает матрицу и выполняет основные
# математические операции, такие как сложение, вычитание, умножение и транспонирование матрицы.
import numpy as np

matrix1 = np.array([[1, 5, 2], [8, 3, 6]])
matrix2 = np.array([[9, 1, 3], [16, 14, 17]])

add_matrix = matrix1 + matrix2
sub_matrix = matrix1 - matrix2
mul_matrix = matrix1 * matrix2
transp_matrix1 = matrix1.transpose()
transp_matrix2 = matrix2.transpose()
print("Addition of matrices:", add_matrix)
print("Subtraction of matrices:", sub_matrix)
print("Multiplication of matrices:", mul_matrix)
print("Transposition of matrice 1:", transp_matrix1)
print("Transposition of matrice 2:", transp_matrix2)

