import pandas as pd
import numpy as np
import re

# Загружаем данные
def download ():
    df = pd.read_csv("core/database/raw_database.csv")
    return df

# Информация о датасете
def check_dataset (df):
    print("Общая информация о датасете")
    print("=" * 40)

    # Размер датасета
    print(f"Количество строк: {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}\n")

    # Названия столбцов
    print("Список столбцов:")
    print(df.columns.tolist(), "\n")

    # Пропущенные значения
    print("Пропущенные значения:")
    print(df.isnull().sum(), "\n")

    # Уникальные значения
    print("Уникальные значения по каждому столбцу:")
    print(df.nunique(), "\n")

    # Типы данных
    print("Типы данных:")
    print(df.dtypes, "\n")

    # Статистика по числовым признакам
    print("Описательная статистика:")
    print(df.describe())

# Предобработка данных
def clear_dataset(df):
    # Удаляем ненужные столбцы
    df = df.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Num_of_Loan', 'Credit_Mix'])

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

    # Заменяем аномальные значения на NaN
    df['Age'] = df['Age'].where(df['Age'].between(18, 100), np.nan)
    df['Annual_Income'] = df['Annual_Income'].where(df['Annual_Income'] > 0, np.nan)
    df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].where(df['Monthly_Inhand_Salary'] > 0, np.nan)
    df['Num_Bank_Accounts'] = df['Num_Bank_Accounts'].where(df['Num_Bank_Accounts'] <= 20, np.nan)
    df['Num_Credit_Card'] = df['Num_Credit_Card'].where(df['Num_Credit_Card'] <= 15, np.nan)
    df['Interest_Rate'] = df['Interest_Rate'].where(df['Interest_Rate'].between(0, 100), np.nan)
    df['Delay_from_due_date'] = df['Delay_from_due_date'].where(df['Delay_from_due_date'].between(0, 365), np.nan)
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].where(df['Num_of_Delayed_Payment'] <= 60, np.nan)
    df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].where(df['Changed_Credit_Limit'].between(-1e6, 1e6), np.nan)
    df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].where(df['Num_Credit_Inquiries'] <= 50, np.nan)
    df['Outstanding_Debt'] = df['Outstanding_Debt'].where(df['Outstanding_Debt'].between(0, 1e6), np.nan)
    df['Credit_Utilization_Ratio'] = df['Credit_Utilization_Ratio'].where(df['Credit_Utilization_Ratio'].between(0, 100), np.nan)
    df['Credit_History_Age'] = df['Credit_History_Age'].where(df['Credit_History_Age'] <= 480, np.nan)
    df['Total_EMI_per_month'] = df['Total_EMI_per_month'].where(df['Total_EMI_per_month'] < 100000, np.nan)
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].where(df['Amount_invested_monthly'].between(0, 50000), np.nan)
    df['Monthly_Balance'] = df['Monthly_Balance'].where(df['Monthly_Balance'].between(-1e6, 1e6), np.nan)

    # Заполняем пропуски медианами
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Кодируем категориальные признаки
    cat_cols = df.select_dtypes(include='object').columns.drop('Credit_Score')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df['Credit_Score'] = df['Credit_Score'].map({'Poor': 0,
                                                'Standard': 1,
                                                'Good': 2})
    
    df.to_csv('core/database/database.csv', index=False)
    return df

def merge_data():

    # Загружаем реальный и синтетические датасеты
    real_df = pd.read_csv('core/database/database.csv')
    synthetic_df_0 = pd.read_csv('core/database/generated_class_0_clean.csv')
    synthetic_df_2 = pd.read_csv('core/database/generated_class_2_clean.csv')

    # Проверяем и приводим порядок столбцов к такому же, как в реальном датасете
    synthetic_df_0 = synthetic_df_0[real_df.columns]
    synthetic_df_2 = synthetic_df_2[real_df.columns]

    # Объединяем все вместе
    combined_df = pd.concat([real_df, synthetic_df_0, synthetic_df_2], axis=0).reset_index(drop=True)

    # Сохраняем в исходный CSV
    combined_df.to_csv('core/database/database.csv', index=False)

    return combined_df

# Запуск
# df = download()
# check_dataset(df)
# df = clear_dataset(df)
# check_dataset(df)
# df = merge_data()
# check_dataset(df)