import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


def check_coefs():
    # Загрузим данные
    df = pd.read_csv("core/database/database.csv")

    # Разделяем на признаки и целевую переменную
    X = df.drop(columns=["Credit_Score"])  # или другую целевую колонку
    y = df["Credit_Score"]

    # Разделим на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабируем признаки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучаем логистическую регрессию
    model = LogisticRegression(max_iter=1000, multi_class="ovr")
    model.fit(X_train_scaled, y_train)

    # Получаем коэффициенты
    coefs = model.coef_
    classes = model.classes_

    # Формируем DataFrame
    for i, class_label in enumerate(classes):
        coef_df = pd.DataFrame({
            "Признак": X.columns,
            f"Вес (Logit) для класса {class_label}": coefs[i]
        })
        print(f"\n==== Класс {class_label} ====")
        print(coef_df.sort_values(by=f"Вес (Logit) для класса {class_label}", key=abs, ascending=False).to_string(index=False))

# Запуск
# check_coefs()