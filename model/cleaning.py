import pandas as pd
import numpy as np
import re

# Загружаем данные
def download ():
    df = pd.read_csv("database/raw_database.csv")
    return df

# Информация о датасете
def check_dataset (df):
    # Атрибуты
    print(df.columns)

    # Проверка на пропущенные значения
    missing_values = df.isnull().sum()
    print(f'Пропущенные значения:\n{missing_values}')

    # Проверка на уникальность
    unique_values = df.nunique()
    print(f'Уникальные значения:\n{unique_values}')

    # Форматы
    print(f'Форматы данных:\n{df.dtypes}')

# Предобработка данных
def clear_dataset(df):
    # Удаляем ненужные столбцы
    df = df.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Num_of_Loan'])

     # Удаляем символы и преобразуем числовые строки
    for col in ['Age', 'Annual_Income', 'Outstanding_Debt',
                'Amount_invested_monthly', 'Monthly_Balance',
                'Num_of_Delayed_Payment', 'Changed_Credit_Limit']:
        df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Преобразуем Credit_History_Age в месяцы
    def parse_credit_age(s):
        if pd.isnull(s) or not isinstance(s, str):
            return np.nan
        match = re.match(r'(\d+)\s+Years.*?(\d+)\s+Months', s)
        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            return years * 12 + months
        return np.nan

    df['Credit_History_Age'] = df['Credit_History_Age'].apply(parse_credit_age)

    # Заполняем пропуски медианами
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Кодируем категориальные признаки
    cat_cols = df.select_dtypes(include='object').columns.drop('Credit_Score')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df['Credit_Score'] = df['Credit_Score'].map({'Poor': 0,
                                                'Standard': 1,
                                                'Good': 2})

    df.to_csv('database/database.csv', index=False)
    return df

# Ручное удаление ошибок
def after_cleaning (df):
    df = df.drop(columns=['Occupation________', 'Credit_Mix__'])
    df.to_csv('database/database.csv', index=False)
    return df


# Запуск
df = download()
check_dataset(df)
df = clear_dataset(df)
check_dataset(df)
df = after_cleaning(df)
check_dataset(df)