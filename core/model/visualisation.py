import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

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
        plt.xlabel(col)
        plt.ylabel('Частота')
        plt.xlim(0, max_val)
        plt.tight_layout()
        plt.show()

# Запуск
# visualise_num()