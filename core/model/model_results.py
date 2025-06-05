import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

def check_model_results():
    # Загрузка данных
    df = pd.read_csv("core/database/database.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    # Делим на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Загрузка модели
    model = joblib.load("core/model/_stacking_model.pkl")

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    class_names = ['Плохой', 'Средний', 'Хороший']
    n_classes = len(class_names)

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Матрица ошибок")
    plt.tight_layout()
    plt.show()

    # ROC-кривые
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые по классам")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Запуск
check_model_results()