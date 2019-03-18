import pytest
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics.classification import accuracy_score
from sklearn.svm import NuSVC

from src.customstransforms import ClfSwitcher

@pytest.fixture
def dataset():
    df_test = pd.read_csv('../../data/interim/test.csv')
    df_train = pd.read_csv('../../data/interim/train.csv')

    y_train = df_train['Survived']
    X_train = df_train.drop(columns='Survived')

    X_test = df_test

    return X_train, X_test, y_train

@pytest.fixture
def pipeline():
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    select_numeric_cols = FunctionTransformer(lambda X: X.select_dtypes(exclude=['object']), validate=False)
    return Pipeline([
        ('select_features', select_numeric_cols),
        ('simple_inputer', imp),
        ('clf', ClfSwitcher())
    ])

def test_hyper_parameter_optmization(dataset, pipeline):
    X_train, X_test, y_train = dataset

    parameters = [
        {
            'clf__estimator': [GradientBoostingClassifier()],
            'clf__estimator__loss': ('deviance', 'exponential'),
            'clf__estimator__learning_rate': (0.15, 0.2),
            'clf__estimator__n_estimators': (500, 600),
        },
    ]

    gscv = GridSearchCV(
        pipeline, parameters, cv=2, n_jobs=12, verbose=3
    )
    gscv.fit(X_train, y_train)
    y = gscv.predict(X_train)
    print(y)
    score = accuracy_score(y_train, y)

    print("Score: " + str(score))
    assert isinstance(score, float)

