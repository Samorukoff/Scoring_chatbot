import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def corr_map():
    # Загрузка датасета
    df = pd.read_csv('core/database/database.csv')

    # Выбираем числовые колонки
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Вычисляем корреляции
    correlations = numeric_df.corr()['Credit_Score'].drop('Credit_Score').sort_values()

    # Визуализация
    plt.figure(figsize=(8, max(6, len(correlations) * 0.4)))  # Авторазмер по количеству признаков
    sns.barplot(
        x=correlations.values,
        y=correlations.index,
        palette=sns.diverging_palette(240, 10, n=correlations.shape[0])
    )

    plt.title('Корреляция признаков с Credit_Score', fontsize=14)
    plt.xlabel('Коэффициент корреляции', fontsize=12)
    plt.ylabel('Признаки', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def visualise_num():
    df = pd.read_csv("core/database/database.csv")

    # Все числовые колонки
    all_numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Цикл по всем числовым колонкам, включая дамми и бинарные
    for col in all_numeric_columns:
        max_val = df[col].max()

        plt.figure(figsize=(min(10, max_val / 10 + 2), 4))  # ширина зависит от max значения
        sns.histplot(data=df, x=col, bins=50, kde=False)

        plt.title(f'Распределение: {col}')
        plt.xlabel('Числовые атрибуты')
        plt.ylabel('Частота')
        plt.xlim(0, max_val)
        plt.tight_layout()
        plt.show()

# Запуск
# corr_map()
# visualise_num()