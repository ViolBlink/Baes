import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

#Начальный день
N_Start = 40

#Послендний дней
N_Last = 100

#Считываем exel таблицу
df = pd.read_excel('Cowid_Russia.xlsx', sheet_name='Лист3')

#Создаем массив случаев и новых случаев соответственно и выбераем данные в заданном периоде
Total_Cases = df['total_cases'][N_Start:N_Last].to_numpy()
New_Cases = df['new_cases'][N_Start:N_Last].to_numpy()

##Срез
# New_Cases = New_Cases[N_Start:N_Last]
# Total_Cases = Total_Cases[N_Start:N_Last]

#Берем лог от данных
Ln_Total_Cases = np.log(Total_Cases)
Ln_New_Cases = []

#Кол-во дней
Days = N_Last - N_Start

for newCase in New_Cases:
    if newCase != 0:
        Ln_New_Cases.append(np.log(newCase))
    else:
        Ln_New_Cases.append(0)

xs = np.array(range(Days)).reshape(-1, 1)

#Создаем полинаминальные признаки
Polinomial = PolynomialFeatures(degree=2, include_bias=False)
Polinomial.fit(xs)

X_P = Polinomial.transform(xs)

model = linear_model.LinearRegression(fit_intercept=True)
model.fit(X_P, Ln_New_Cases)

x = range(Days)

#plt.plot(x, model.coef_[1] * x * x + model.coef_[0] * x + model.intercept_)
#plt.plot(x, model.predict(X_P))

New_Cases_Pred = np.exp(model.predict(X_P))

plt.plot(x, New_Cases_Pred)

plt.plot(range(Days), New_Cases)

plt.grid(True)
plt.show()