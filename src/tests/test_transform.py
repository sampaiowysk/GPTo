import pytest
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from src.pipeline.transform import get_standard_scale_with_pca_etl

@pytest.fixture
def dataset():
    data = load_boston()
    return (
        pd.DataFrame(data['data'], columns=data.feature_names),
        pd.Series(data['target'], name='PRICE')
    )

@pytest.fixture
def sample(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

def test_transform_etl_with_ridge (sample):
    pipe = get_standard_scale_with_pca_etl()
    pipe.steps.append(('regressor', Ridge()))

    X_train, X_test, y_train, y_test = sample
    pipe = pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    assert isinstance(score, float)
    assert score > 0.6
