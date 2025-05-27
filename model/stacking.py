import pandas as pd

def check_dataset ():
    df = pd.read_csv('database/train.csv')
    df = df.drop(columns=['ID', 'Customer_ID', 'Name'])
    # Проверка на пропущенные значения
    missing_values = df.isnull().sum()
    print(f'Пропущенные значения:\n{missing_values}')
    # Проверка на уникальность
    unique_values = df.nunique()
    print(f'Уникальные значения:\n{unique_values}')

check_dataset()