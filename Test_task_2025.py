import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from tabulate import tabulate
import matplotlib.pyplot as plt
from openpyxl.utils import get_column_letter
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor


#Этап 1
#Используемые функци
def Mat_Exp(df, column_name):
    return df[column_name].mean()


def show_y_predicted(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    
    # Создаем индексы для оси X
    indexes = np.arange(len(y_true))
    
    # Фактические значения (синие линии)
    plt.plot(indexes, y_true, color='blue', 
             markersize=6, linewidth=1.5, 
             label='Фактические значения')
    
    # Предсказанные значения (красные линии)
    plt.plot(indexes, y_pred, color='red', 
             markersize=8, linewidth=1.5, 
             label='Предсказанные значения')
    
    # Настройки графика
    plt.xlabel('Номер наблюдения', fontsize=12)
    plt.ylabel('Доход, руб.', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Добавляем легенду
    plt.legend(fontsize=10, frameon=True, 
               shadow=True, loc='best')
    
    # Улучшаем отображение
    plt.tight_layout()
    plt.show()



def QQPlots(df):
    fig = sm.qqplot(df["201710"], line = "q")
    plt.title("qq_plot: Октябрь 2017")
    plt.show()
    fig = sm.qqplot(df["201711"], line = "q")
    plt.title("qq_plot: Ноябрь 2017")
    plt.show()
    fig = sm.qqplot(df["201712"], line = "q")
    plt.title("qq_plot: Декабрь 2017")
    plt.show()
    fig = sm.qqplot(df["202010"], line = "q")
    plt.title("qq_plot: Октябрь 2020")
    plt.show()
    fig = sm.qqplot(df["202011"], line = "q")
    plt.title("qq_plot: Ноябрь 2020")
    plt.show()
    fig = sm.qqplot(df["202012"], line = "q")
    plt.title("qq_plot: Декабрь 2020")
    plt.show()



    

#Этап 2
# Читаем XLSB
with pd.ExcelFile("Test_task_2025.xlsb", engine="pyxlsb") as xlsb:
    # Сохраняем каждый лист в CSV
    for sheet in xlsb.sheet_names:
        pd.read_excel(xlsb, sheet).to_csv(f"{sheet}.csv", index=False)

print("Конвертация завершена!")


#Этап 3
#Настройки
pd.set_option('display.width', 200)  # Ширина вывода 200 символов
pd.set_option('display.max_columns', 10)  # Показывать до 10 столбцов в строке  
pd.set_option('display.max_rows', None)# Установка максимального числа строк для вывода

#Этап 4
#Подготовка данных
df = pd.read_csv("Выгрузка .csv")
print(df.columns.tolist(),'\n')  # Выведет все названия столбцов
# Удаляем все пробелы в названиях столбцов
df.columns = df.columns.str.replace(' ', '')

#Визуализация распределения до удаления пропусков и общего итога
#sns.boxplot(data=df[['202012']])
#plt.title("Распределение до удаления пропусков и общего итога")
#plt.show()
#QQPlots(df)

# Удаление Общего итога
df = df[:-1]
df = df[~df['REGION'].str.contains('Общий итог', na=False)]
print('Последние строки таблицы:')
print(df.tail(),'\n') #Проверяем, что итога нет

#Этап 5
#Разведочный анализ (EDA)
#1. Осмотр данных
print("\nВерхние строки таблицы:")
print(df.head())
print("\nРазмер таблицы:")
print(df.shape)
print("\nИнформация об столбцах таблицы:")
print(df.info())
print("\nИнформация об пустых значениях:")
print(df.isnull().sum())
print("\nИнформация о дубликатах:")
print(df.duplicated().sum(),'\n')

print("\nИнформация о данных таблицы до обработки:")
print(df.describe().round(2))
print("\nМат.ожидание за период 2017_10 = ", Mat_Exp(df,"201710"))
print("\nМат.ожидание за период 2017_11 = ", Mat_Exp(df,"201711"))
print("\nМат.ожидание за период 2017_12 = ", Mat_Exp(df,"201712"))
print("\nМат.ожидание за период 2020_10 = ", Mat_Exp(df,"202010"))
print("\nМат.ожидание за период 2020_11 = ", Mat_Exp(df,"202011"))
print("\nМат.ожидание за период 2020_12 = ", Mat_Exp(df,"202012"))
print("\nДисперсия\n\n",df.var(numeric_only=True),'\n')
#Матрица корреляции
print("\nМатрица корреляции\n", df.corr(numeric_only=True),'\n')


#Визуализация распределения до очистки
sns.boxplot(data=df[['202012']])
plt.title("Распределение до удаления пропусков")
plt.show()

QQPlots(df)




#Этап 6
# Работа с пропущенными значениями
# Для столбца 'service_group'
null_service_group = df[df['service_group'].isnull()].index
print("Индексы строк с пропущенной услугой:", null_service_group.tolist())

# Список индексов с пропусками
null_indices = [1, 1989, 3619, 4942, 8108, 9214, 9320, 9463, 10861, 11106, 12072]

# Сохраняем в CSV
df.loc[null_indices].to_csv('Данные с пропусками.csv', index=True, encoding='utf-8-sig')
print("Сохранено в файл: Данные с пропусками.csv",'\n')

#Удаление строк с пропусками через индексы
df_cleaned = df.drop(index=null_indices)
print(f"Было: {len(df)} строк, стало: {len(df_cleaned)}")
print("Оставшиеся пропуски:")
print(df_cleaned.isnull().sum(),'\n')

#Применение критерия Колмогорова-Смирнова (для проверки, повлияло ли удаление строк с пропусками или нет)
# Выбираем числовой столбец для сравнения (например, '202012')
original_col = df['202012'].dropna()  # Исходные данные (без NaN)
cleaned_col = df_cleaned['202012']    # Очищенные данные

ks_stat, p_value = ks_2samp(original_col, cleaned_col)
print(f"K-S статистика: {ks_stat:.4f}, p-value: {p_value:.4f}")

if p_value > 0.05:
    print("Распределения НЕ различаются (p > 0.05)")
else:   
    print("Распределения РАЗЛИЧАЮТСЯ (p ≤ 0.05)")



print("\nИнформация о данных таблицы ПОСЛЕ ОЧИСТКИ:")
print(df_cleaned.describe().round(2))
print("\nМат.ожидание за период 2017_10 = ", Mat_Exp(df_cleaned,"201710"))
print("\nМат.ожидание за период 2017_11 = ", Mat_Exp(df_cleaned,"201711"))
print("\nМат.ожидание за период 2017_12 = ", Mat_Exp(df_cleaned,"201712"))
print("\nМат.ожидание за период 2020_10 = ", Mat_Exp(df_cleaned,"202010"))
print("\nМат.ожидание за период 2020_11 = ", Mat_Exp(df_cleaned,"202011"))
print("\nМат.ожидание за период 2020_12 = ", Mat_Exp(df_cleaned,"202012"))
print("\nДисперсия\n\n",df_cleaned.var(numeric_only=True),'\n')
#Матрица корреляции
print("\nМатрица корреляции\n", df_cleaned.corr(numeric_only=True),'\n')

#Визуальное представление матрицы корреляции
numeric_cols = ['201710', '201711', '201712', '202010', '202011', '202012']
sns.heatmap(df_cleaned[numeric_cols].corr(), cmap = 'BrBG',linewidths = 2, annot=True)
plt.show()

'''
# Расчет относительного изменения
mean_change = (df_cleaned.mean() - df.mean()) / df.mean() * 100
var_change = (df_cleaned.var() - df.var()) / df.var() * 100
print(f"Среднее изменение матожидания: {mean_change.mean():.2f}%")
print(f"Среднее изменение дисперсии: {var_change.mean():.2f}%\n")


#Визуализация распределения после очистки
sns.boxplot(data=df_cleaned[['202012']])
plt.title("Распределение после удаления итогов и пропусков")
plt.show()

QQPlots(df_cleaned)
'''

#Сохранение очищенной выборки
df_cleaned.to_csv('cleaned_data.csv', index=False)

#Этап 7
# 1.Сравнение доходности тарифных групп по периодам
# Доходы по тарифам за 2017
print('Доходы по тарифам за 2017')
periods_17 = ['201710', '201711', '201712']

# Группируем и агрегируем данные по всем периодам
result = (
    df_cleaned.groupby('TP_GROUP')[periods_17]
    .sum()
    .loc[lambda x: x.sum(axis=1) > 0]  # Фильтр ненулевых групп
    .sort_values(by='201712', ascending=False)  # Сортировка по последнему периоду
)
# Форматируем числа и переименовываем столбцы
formatted = result.applymap(lambda x: f"{x:,.2f} руб." if x > 0 else "-")
formatted.columns = [f"{p[:4]}-{p[4:]}" for p in periods_17]  # Переименовываем в "ГГГГ-ММ"

# Выводим красивую таблицу
print(tabulate(
    formatted.reset_index(),
    headers='keys',
    tablefmt='grid',
    showindex=False,
    stralign='left',
    numalign='right'
),'\n')


#Доходы по тарифам за 2020
print('Доходы по тарифам за 2020')
# Периоды за 2020 для анализа(должны существовать в DataFrame как столбцы)
periods_20 = ['202010', '202011', '202012']  


# Группируем и агрегируем данные по всем периодам
result = (
    df_cleaned.groupby('TP_GROUP')[periods_20]
    .sum()
    .loc[lambda x: x.sum(axis=1) > 0]  # Фильтр ненулевых групп
    .sort_values(by='202012', ascending=False)  # Сортировка по последнему периоду
)

# Форматируем числа и переименовываем столбцы
formatted = result.applymap(lambda x: f"{x:,.2f} руб." if x > 0 else "-")
formatted.columns = [f"{p[:4]}-{p[4:]}" for p in periods_20]  # Переименовываем в "ГГГГ-ММ"

# Выводим красивую таблицу
print(tabulate(
    formatted.reset_index(),
    headers='keys',
    tablefmt='grid',
    showindex=False,
    stralign='left',
    numalign='right'
),'\n')



#2. Общая сводка за 2017 и 2020 + общий доход
# Подготовка данных
periods = ['201710', '201711', '201712', '202010', '202011', '202012']  # Все периоды

# Группировка с сохранением всех TP_GROUP (даже с нулевыми доходами)
grouped = df_cleaned.groupby('TP_GROUP')[periods].sum().sort_values(by='202012', ascending=False)

# Добавляем строку "Общий доход"
total_row = pd.DataFrame(
    data=[['Общий доход'] + [grouped[col].sum() for col in periods]],
    columns=['TP_GROUP'] + periods
).set_index('TP_GROUP')

# Объединяем с основными данными
final_table = pd.concat([grouped, total_row])

# Форматирование (рубли и переименование столбцов)
formatted = (
    final_table
    .applymap(lambda x: f"{x:,.2f} руб." if x > 0 else "0.00 руб.")
)

# Сохранение в Excel с настройкой ширины столбцов
with pd.ExcelWriter('Доходы_по_месяцам.xlsx', engine='openpyxl') as writer:
    formatted.reset_index().to_excel(writer, sheet_name='Доходы', index=False)
    
    # Получаем доступ к листу
    worksheet = writer.sheets['Доходы']
    
    # Настраиваем ширину столбцов
    for column in worksheet.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        
        for cell in column:
            try:
                cell_length = len(str(cell.value))
                if cell_length > max_length:
                    max_length = cell_length
            except:
                pass
        
        # Устанавливаем ширину с небольшим запасом
        adjusted_width = (max_length + 2) * 1.2
        worksheet.column_dimensions[column_letter].width = min(adjusted_width, 30)  # Ограничиваем максимальную ширину
    
print('Файл сохранен')

#Этап 8
# Подготовка данных для построения сводных таблиц
# Переводим столбцы с датами в строки
# Удаляем строки, где 'Регион' = 'Общий итог'
df_melted = df_cleaned.melt(
    id_vars=['REGION', 'TP_GROUP', 'service_group'],
    value_vars=['201710', '201711', '201712', '202010', '202011', '202012'],
    var_name='DATE',
    value_name='REVENUE'
)

# Удаляем строки, где 'Регион' = 'Общий итог'
#df_melted = df_melted[df_melted['REGION'] != 'Общий итог'] 
# Разделяем дату на год и месяц
df_melted['Год'] = df_melted['DATE'].str[:4]  # 201710 → 2017
df_melted['MONTH'] = df_melted['DATE'].str[-2:]  # 201710 → 10

month_names = {
    '10': 'Октябрь',
    '11': 'Ноябрь',
    '12': 'Декабрь'
}


# Переименовываем столбец
df_melted['Месяц'] = df_melted['MONTH'].map(month_names)
df_melted = df_melted.rename(columns={'REVENUE': 'Доход'})
df_melted = df_melted.rename(columns={'TP_GROUP': 'Тариф'}) 
df_melted = df_melted.rename(columns={'REGION': 'Регион'})
df_melted['Доход'] = df_melted['Доход'] / 1_000_000  # Переводим в миллионы
pd.options.display.float_format = '{:,.1f} млн'.format



########################
#Этап 9
#1. Анализ по годам:
#a) Сравнение доходов по годам: 2017 vs 2020 (общий доход)
total_by_year = df_melted.groupby('Год')['Доход'].sum()
print(total_by_year,'\n')


#b) Доходы по месяцам в разрезе лет:
pivot_month_year = df_melted.pivot_table(
    index='Месяц',
    columns='Год',
    values='Доход',
    aggfunc='sum'
)
print(pivot_month_year,'\n')

# График по месяцам
pivot_month_year.plot(
    kind='line',
    marker='o',
    title='Доходы по месяцам: 2017 vs 2020'
)
plt.show()


#2. Анализ по регионам
#a) Топ регионов по доходу за 2017 год:
print('Топ регионов по доходу за 2017 год:')
regions_2017 = df_melted[df_melted['Год'] == '2017'] \
    .groupby('Регион')['Доход'].sum() \
    .sort_values(ascending=False)
#regions_2017.plot(kind='pie', autopct='%1.1f%%')
print(regions_2017.to_string(header=False),'\n')

# Круговая диаграмма 2017
regions_2017.plot(
    kind='pie',
    autopct='%1.f%%',
    title='Доля регионов в общем доходе (2017)',
    figsize=(8, 8)
)
plt.ylabel('')  # Убираем подпись оси Y
plt.show()


#b) Топ регионов по доходу за 2020 год:
print('Топ регионов по доходу за 2020 год:')
regions_2020 = df_melted[df_melted['Год'] == '2020'] \
    .groupby('Регион')['Доход'].sum() \
    .sort_values(ascending=False)
#regions_2020.plot(kind='pie', autopct='%1.1f%%')
print(regions_2020.to_string(header=False), '\n')

# Круговая диаграмма
regions_2020.plot(
    kind='pie',
    autopct='%1.1f%%',
    title='Доля регионов в общем доходе (2020)',
    figsize=(8, 8)
)
plt.ylabel('')  # Убираем подпись оси Y
plt.show()



#c)Сравнение регионов в 2017 и 2020:
pivot_region_year = df_melted.pivot_table(
    index='Регион',
    columns='Год',
    values='Доход',
    aggfunc='sum'
)
pivot_region_year['Доход/убыток'] = (pivot_region_year['2020'] - pivot_region_year['2017'] )
# Расчет процентов
pivot_region_year['Изменение, %'] = ((pivot_region_year['2020'] - pivot_region_year['2017']) / pivot_region_year['2017'] * 100).round(1)
# Убираем " млн" и преобразуем в числа
pd.reset_option('display.float_format')
pivot_region_year['2017'] = pivot_region_year['2017'].map('{:,.1f} млн'.format)
pivot_region_year['2020'] = pivot_region_year['2020'].map('{:,.1f} млн'.format)
pivot_region_year['Доход/убыток'] = pivot_region_year['Доход/убыток'].map('{:,.1f} млн'.format)
pivot_region_year['Изменение, %'] = pivot_region_year['Изменение, %'].astype(str) + '%'
print(pivot_region_year,'\n')

pd.reset_option('display.float_format')

# Сумма по регионам (без "Общего итога")
sum_regions_2017 = pivot_region_year.loc[['Регион 1', 'Регион 2', 'Регион 3'], '2017'].str.replace(' млн', '').astype(float).sum()
sum_regions_2020 = pivot_region_year.loc[['Регион 1', 'Регион 2', 'Регион 3'], '2020'].str.replace(' млн', '').astype(float).sum()

print("Сумма регионов 2017:", sum_regions_2017)
print("Сумма регионов 2020:", sum_regions_2020)


#3. Анализ по тарифам (TP_GROUP)
#a) Топ-3 тарифных групп по доходу (2017):
top_tariffs_2017 = df_melted[df_melted['Год'] == '2017'] \
    .groupby('Тариф')['Доход'].sum() \
    .nlargest(3)

# Строим график с настройками
ax = top_tariffs_2017.plot(
    kind='barh',
    title='Топ тарифов в 2017 году',
    figsize=(10, 5))
    
# Добавляем подписи осей
ax.set_xlabel('Доход (млн. руб.)')  # Подпись оси X
ax.set_ylabel('Тариф')              # Подпись оси Y

# Добавляем значения на столбцы
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=3)

# Выводим таблицу и график
formatted = top_tariffs_2017.map('{:,.1f} млн.'.format)
print('Топ 3 тарифа за 2017:')
print(formatted.to_frame(name='Доход').to_string(header=True),'\n')
plt.tight_layout()  # Оптимизация расположения элементов
plt.show()


#b) Топ-3 тарифных групп по доходу (2020):
top_tariffs_2020 = df_melted[df_melted['Год'] == '2020'] \
    .groupby('Тариф')['Доход'].sum() \
    .nlargest(3)

# Строим график с настройками
ax = top_tariffs_2020.plot(
    kind='barh',
    title='Топ тарифов в 2020 году',
    figsize=(10, 5))
    
# Добавляем подписи осей
ax.set_xlabel('Доход (млн. руб.)')  # Подпись оси X
ax.set_ylabel('Тариф')              # Подпись оси Y

# Добавляем значения на столбцы
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=3)

# Выводим таблицу и график
formatted = top_tariffs_2020.map('{:,.1f} млн.'.format)
print('Топ 3 тарифа за 2020:')
print(formatted.to_frame(name='Доход').to_string(header=True),'\n')
plt.tight_layout()  # Оптимизация расположения элементов
plt.show()


#c) Сравнение тарифов между 2017 и 2020:
pivot_tariff = df_melted.pivot_table(
    index='Тариф',
    columns='Год',
    values='Доход',
    aggfunc='sum'
).dropna()
# Добавляем разницу (в млн)
pivot_tariff['Разница'] = pivot_tariff['2020'] - pivot_tariff['2017']

# Преобразуем в тысячи
pivot_tariff[['2017', '2020', 'Разница']] = pivot_tariff[['2017', '2020', 'Разница']] * 1000

# Сортировка и вывод
pivot_tariff = pivot_tariff.sort_values('Разница', ascending=False)
print('Сравнение тарифов между 2017 и 2020 (доход (тыс.руб.))')
print(pivot_tariff.round(0).astype(int),'\n')  # Округление до целых


#4. Анализ по услугам
print('Доход от услуги "услуга_1"')
internet_revenue = df_melted[df_melted['service_group'] == 'услуга_1'] \
    .groupby(['Год', 'Месяц'])['Доход'].sum() \
    .unstack()*1000 #настройка чтоб выводились тыс. вместо млн.
pd.options.display.float_format = '{:,.0f} тыс.'.format
print(internet_revenue,'\n')


# Построение графика
ax = internet_revenue.plot(
    kind='bar',
    title='Доходы от "услуга_1"',
    figsize=(12, 6),
    xlabel='Год',
    ylabel='Сумма дохода (тыс. руб.)',
    rot=0
    )
    
# Добавление подписей значений
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', labels=[f'{x:,.0f} тыс.' for x in container.datavalues])
    
plt.tight_layout()
plt.show()







#Этап 10 ML Предсказание на декабрь 2017 для услуги услуга_1
df_melted['Доход'] = df_melted['Доход'] * 1_000_000  # Переводим из миллионов в исходные единицы
df_melted = df_melted[df_melted['service_group'] == 'услуга_1'] #фильтруем по услуге

# Разделение на обучающую и тестовую выборки
train = df_melted[(df_melted['Год'] == '2017') & (df_melted['MONTH'].isin(['10', '11']))]
test = df_melted[(df_melted['Год'] == '2017') & (df_melted['MONTH'] == '12')]


# Подготовка признаков
features = ['Регион', 'Тариф', 'service_group', 'Месяц']
target = 'Доход'


# Преобразование категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), features)
    ])

X_train = preprocessor.fit_transform(train[features])
Y_train = train[target]
X_test = preprocessor.transform(test[features])
Y_test = test[target]

# Обучение модели
model = LinearRegression()
model.fit(X_train, Y_train)


# Прогнозирование
Y_pred = model.predict(X_test)

# Оценка модели
print('Метрики для LinearRegression')
print(f"Размер обучающей выборки: {len(train)} записей")
print(f"Размер тестовой выборки: {len(test)} записей")
print(f"MAE: {mean_absolute_error(Y_test, Y_pred):.1f}")
print(f"MSE: {mean_squared_error(Y_test, Y_pred):.1f}")
print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, Y_pred)):.1f}")
print(f"R2 Score: {r2_score(Y_test, Y_pred):.3f}\n")



# Визуализация важности коэффициентов (для линейной регрессии)
show_y_predicted(Y_test,Y_pred,"Линейная регрессия")

plt.figure(figsize=(10, 6))

# Факт vs Прогноз (синие точки)
plt.scatter(Y_test, Y_pred, alpha=0.6, c='blue', edgecolor='k', 
            label='Прогноз vs Факт')

# Линия идеального прогноза (красная пунктирная)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 
         'r--', label='Идеальная точность (y = x)')

plt.title('Сравнение прогноза и фактического дохода\n"Доп.интернет" (Декабрь 2017)')
plt.xlabel('Фактический доход (руб)')
plt.ylabel('Прогнозируемый доход (руб)\n')
plt.grid(True, linestyle='--', alpha=0.7)

# Улучшенная легенда
plt.legend(loc='upper left', fontsize=10, framealpha=1)

# Добавим аннотации для ясности
plt.text(0.7*max(Y_test), 0.1*max(Y_test), 
         'Прогноз занижен', color='red', fontsize=10)
plt.text(0.1*max(Y_test), 0.7*max(Y_test), 
         'Прогноз завышен', color='red', fontsize=10)

plt.show()




# Пример проверки улучшений
# Случайный лес
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('Метрики для RandomForestRegressor')
print(f"MAE: {mean_absolute_error(Y_test, Y_pred):.1f}")
print(f"MSE: {mean_squared_error(Y_test, Y_pred):.1f}")
print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, Y_pred)):.1f}")
print(f"R2 Score: {r2_score(Y_test, Y_pred):.3f}\n")


show_y_predicted(Y_test,Y_pred,"Случайный лес")



plt.figure(figsize=(10, 6))

# Факт vs Прогноз (синие точки)
plt.scatter(Y_test, Y_pred, alpha=0.6, c='blue', edgecolor='k', 
            label='Прогноз vs Факт')

# Линия идеального прогноза (красная пунктирная)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 
         'r--', label='Идеальная точность (y = x)')

plt.title('Сравнение прогноза и фактического дохода\n"услуга_1" (Декабрь 2017)')
plt.xlabel('Фактический доход (руб)')
plt.ylabel('Прогнозируемый доход (руб)')
plt.grid(True, linestyle='--', alpha=0.7)

# Улучшенная легенда
plt.legend(loc='upper left', fontsize=10, framealpha=1)

# Добавим аннотации для ясности
plt.text(0.7*max(Y_test), 0.1*max(Y_test), 
         'Прогноз занижен', color='red', fontsize=10)
plt.text(0.1*max(Y_test), 0.7*max(Y_test), 
         'Прогноз завышен', color='red', fontsize=10)

plt.show()


# CatBoostRegressor для регрессии - библиотека градиентного бустинга
model = CatBoostRegressor(n_estimators=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('Метрики для CatBoost')
print(f"MAE: {mean_absolute_error(Y_test, Y_pred):.1f}")
print(f"MSE: {mean_squared_error(Y_test, Y_pred):.1f}")
print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, Y_pred)):.1f}")
print(f"R2 Score: {r2_score(Y_test, Y_pred):.3f}\n")


show_y_predicted(Y_test,Y_pred,"CatBoost")



plt.figure(figsize=(10, 6))

# Факт vs Прогноз (синие точки)
plt.scatter(Y_test, Y_pred, alpha=0.6, c='blue', edgecolor='k', 
            label='Прогноз vs Факт')

# Линия идеального прогноза (красная пунктирная)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 
         'r--', label='Идеальная точность (y = x)')

plt.title('Сравнение прогноза и фактического дохода\n"услуга_1" (Декабрь 2017)')
plt.xlabel('Фактический доход (руб)')
plt.ylabel('Прогнозируемый доход (руб)')
plt.grid(True, linestyle='--', alpha=0.7)

# Улучшенная легенда
plt.legend(loc='upper left', fontsize=10, framealpha=1)

# Добавим аннотации для ясности
plt.text(0.7*max(Y_test), 0.1*max(Y_test), 
         'Прогноз занижен', color='red', fontsize=10)
plt.text(0.1*max(Y_test), 0.7*max(Y_test), 
         'Прогноз завышен', color='red', fontsize=10)

plt.show()

