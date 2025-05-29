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

def download_data():
    # Загрузка данных
    logging.info("Загрузка данных...")
    df = pd.read_csv("core/database/database.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    # Разделение на обучающую и тестовую выборки
    logging.info("Разделение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def create_model(X_train, X_test, y_train, y_test):
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
            criterion='gini',
            splitter='best',
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=1,
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
        estimator=DecisionTreeClassifier(class_weight='balanced'),
        n_estimators=50,
        max_samples=0.8,
        max_features=0.5,
        bootstrap=False,
        random_state=42
    )

    # Объединяем все в один стек
    logging.info("Создание стекинг-модели...")
    stacking = StackingClassifier(
        estimators=base_models + [('gb', gb), ('bag', bag)],
        final_estimator=LogisticRegression(max_iter=1000),
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
    coefs = stacking.final_estimator_.coef_[0]

    # Формируем имена признаков по моделям
    feature_names = []
    for name, model in stacking.named_estimators_.items():
        if hasattr(model, "predict_proba"):
            n_classes = model.predict_proba(X_test).shape[1]
            for i in range(n_classes):
                feature_names.append(f"{name}_class_{i}")
        else:
            feature_names.append(f"{name}_pred")

    # Создаём DataFrame для удобного отображения
    weights_df = pd.DataFrame({
        'Признак': feature_names,
        'Вес (Logit)': coefs
    }).sort_values(by='Вес (Logit)', key=abs, ascending=False)

    # Выводим
    print(weights_df.to_string(index=False))

    # Выгрузка модели
    logging.info("Сохранение модели в файл 'core/model/_stacking_model.pkl'...")
    joblib.dump(stacking, 'core/model/_stacking_model.pkl')
    
    return stacking

# Запуск
X_train, X_test, y_train, y_test = download_data()
create_model(X_train, X_test, y_train, y_test)

