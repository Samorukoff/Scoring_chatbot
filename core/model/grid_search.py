import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier

def grid_search():

    df = pd.read_csv("core/database/database.csv")
    X = df.drop(columns=['Credit_Score'])
    y = df['Credit_Score']

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # К-ближайших соседей
    knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
    knn_params = {
        'kneighborsclassifier__n_neighbors': [3, 5, 7],
        'kneighborsclassifier__weights': ['uniform', 'distance'],
        'kneighborsclassifier__p': [1, 2],
    }
    knn_search = GridSearchCV(knn_pipe, knn_params, cv=3, n_jobs=-1, verbose=1)
    knn_search.fit(X_train, y_train)
    print("Best KNN:", knn_search.best_params_)

    # Логистическая регрессия
    logreg_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg_params = {
        'logisticregression__C': [0.1, 1.0, 10],
        'logisticregression__solver': ['lbfgs', 'saga'],
        'logisticregression__penalty': ['l2'],
        'logisticregression__class_weight': [None, 'balanced']
    }
    logreg_search = GridSearchCV(logreg_pipe, logreg_params, cv=3, n_jobs=-1, verbose=1)
    logreg_search.fit(X_train, y_train)
    print("Best LogisticRegression:", logreg_search.best_params_)

    # Дерево решений
    dt = DecisionTreeClassifier()
    dt_params = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5],
        'class_weight': [None, 'balanced']
    }
    dt_search = GridSearchCV(dt, dt_params, cv=3, n_jobs=-1, verbose=1)
    dt_search.fit(X_train, y_train)
    print("Best DecisionTree:", dt_search.best_params_)

    # Бустигш (градиентный спуск)
    gb = GradientBoostingClassifier()
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    gb_search = GridSearchCV(gb, gb_params, cv=3, n_jobs=-1, verbose=1)
    gb_search.fit(X_train, y_train)
    print("Best GradientBoosting:", gb_search.best_params_)

    # Бэгинг (дерево решений)
    bag = BaggingClassifier(estimator=DecisionTreeClassifier())
    bag_params = {
        'n_estimators': [10, 50],
        'max_samples': [0.5, 0.8],
        'max_features': [0.5, 0.8],
        'bootstrap': [True, False]
    }
    bag_search = GridSearchCV(bag, bag_params, cv=3, n_jobs=-1, verbose=1)
    bag_search.fit(X_train, y_train)
    print("Best BaggingClassifier:", bag_search.best_params_)

# Запуск
# grid_search()