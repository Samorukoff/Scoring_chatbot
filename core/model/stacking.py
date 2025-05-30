import logging
import joblib

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, StackingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)

def create_model():
    # Загрузка данных
    logging.info("Загрузка данных...")
    df = pd.read_csv("core/database/database.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    # Разделение на обучающую и тестовую выборки
    logging.info("Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Базовые модели
    logging.info("Инициализация базовых моделей...")
    base_models = [
        ('knn', make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                p=1,
            ))),

        ('logreg', make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                penalty='l2',
                C=0.1
            ))),

        ('dt', DecisionTreeClassifier(
            max_depth=7,
            min_samples_split=2,
            min_samples_leaf=5,
            class_weight=None
        ))
    ]

    # Бэгинг и бустинг
    logging.info("Создание ансамблей: бустинг и бэггинг...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=7,
                                         min_samples_split=2,
                                         min_samples_leaf=5,
                                         class_weight=None),
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=False,
        random_state=42
    )

    # Объединяем все в один стек
    logging.info("Создание стекинг-модели...")
    stacking = StackingClassifier(
        estimators=base_models + [('gb', gb), ('bag', bag)],
        final_estimator=LogisticRegression(max_iter=1000,
                                           class_weight='balanced',
                                           solver='lbfgs',
                                           penalty='l2',
                                           C=0.1),
        passthrough=False,
        cv=5,
        n_jobs=-1
    )

    # Обучение стекинг-модели
    logging.info("Обучение стекинг-модели...")
    stacking.fit(X_train, y_train)

    logging.info("Предсказание на тестовой выборке...")
    y_pred = stacking.predict(X_test)

    # Оценка качества
    logging.info("Оценка модели:")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Получаем веса (коэффициенты) LogisticRegression
    importances = stacking.final_estimator_.feature_importances_
    print(importances)

    # Выгрузка модели
    logging.info("Сохранение модели в файл 'core/model/_stacking_model.pkl'...")
    joblib.dump(stacking, 'core/model/_stacking_model.pkl')

    # Порядок переменных для дальнейшей работы
    joblib.dump(X_train.columns.tolist(), "core/model/_feature_order.pkl")
    
    return stacking

# Запуск
create_model()

