from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, label_binarize

#Начальный день обучения
N_StartOfTraining = 40

#Послендний дней обучения
N_EndOfTraining = 100

#Последний день предсказания
N_EndOfPrediction = N_EndOfTraining + 50

#Считываем exel таблицу
df = pd.read_excel('Cowid_Russia.xlsx', sheet_name='Лист3')

#Создаем массив случаев и новых случаев соответственно и выбераем данные в заданном периоде
Total_Cases = df['total_cases'][N_StartOfTraining:N_EndOfTraining].to_numpy()
New_Cases_Linear = df['new_cases'][N_StartOfTraining:N_EndOfTraining].to_numpy()
#Общие и новые случае для предсказаний
Total_Cases_Ridge = df['total_cases'][N_StartOfTraining:N_EndOfPrediction].to_numpy()
New_Cases_Ridge = df['new_cases'][N_StartOfTraining:N_EndOfPrediction].to_numpy()

#Берем лог от данных
Ln_Total_Cases = np.log(Total_Cases)
Ln_New_Cases = []

#Кол-во дней
Days = N_EndOfTraining - N_StartOfTraining

for newCase in New_Cases_Linear:
    if newCase != 0:
        Ln_New_Cases.append(np.log(newCase))
    else:
        Ln_New_Cases.append(0)

#Берем лог от данных для предсказаний
Ln_New_Cases_Ridge = []

for newCase in New_Cases_Ridge:
    if newCase != 0:
        Ln_New_Cases_Ridge.append(np.log(newCase))
    else:
        Ln_New_Cases_Ridge.append(0)

xs = np.array(range(Days)).reshape(-1, 1)

#Создаем полинаминальные признаки
Polinomial = PolynomialFeatures(degree=2, include_bias=False)
Polinomial.fit(xs)

X_P = Polinomial.transform(xs)

#Обучаем линейную регресию
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(X_P, Ln_New_Cases)

x = range(Days)

#plt.plot(x, model.coef_[1] * x * x + model.coef_[0] * x + model.intercept_)
#plt.plot(x, model.predict(X_P))

New_Cases_Pred = np.exp(model.predict(X_P))

#Создание граффика
fig1 = plt.figure(figsize=(8, 8),dpi=100)

ax1 = fig1.add_subplot(111)

ax1.plot(x, New_Cases_Pred, label='Предсказания')

ax1.plot(x, New_Cases_Linear, label='Данные')

ax1.grid(True)
plt.legend()

# Ридж регресия
alpha = np.arange(10, 91, 10)

#Новый граффик
fig2 = plt.figure(figsize=(8, 8),dpi=100)
ax2 = fig2.add_subplot(111)

#Точки для предскозаний 
x = range(N_EndOfPrediction - N_StartOfTraining)

xs = np.array(range(N_EndOfTraining - N_StartOfTraining)).reshape(-1, 1)
xs1 = np.array(range(N_EndOfPrediction - N_StartOfTraining)).reshape(-1, 1)

#Создаем полинаминальные признаки
Polinomial = PolynomialFeatures(degree=2, include_bias=False)
Polinomial.fit(xs)
Polinomial.fit(xs1)

X_P = Polinomial.transform(xs)
X_P1 = Polinomial.transform(xs1)

# Обучаем ридж
for al in alpha:
    model1 = linear_model.Ridge(alpha=al, fit_intercept=True).fit(X_P, Ln_New_Cases)
    ax2.plot(x, np.exp(model1.predict(X_P1)), label="$\\alpha=%f$" % al)

    # Усредняем предсказание
    if al == alpha[0]:
        PredModel = np.exp(model1.predict(X_P1))
    else:
        PredModel = PredModel + np.exp(model1.predict(X_P1))

PredModel = PredModel/len(alpha)

ax2.plot(x, New_Cases_Ridge, label='Данные')
plt.legend()
ax2.grid(True)

fig3 = plt.figure(figsize=(8, 8),dpi=100)
ax3 = fig3.add_subplot(111)

ax3.plot(x, Total_Cases_Ridge, label='Данные')

# Чилсенное интегрирование

IntPredModel = []

for y in x:
    IntPredModel.append(np.sum(PredModel[:y]))
ax3.plot(x, IntPredModel, label='Усреднённое предсказание')

plt.legend()
ax3.grid(True)

plt.show()