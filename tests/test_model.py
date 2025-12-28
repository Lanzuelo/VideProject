import pytest
import pandas as pd
import os
from src.model import DifficultyClassifier
from src.data_gen import generate_synthetic_data

@pytest.fixture
def sample_data():
    # Создаем временные данные для тестов
    if not os.path.exists('data'):
        os.makedirs('data')
    generate_synthetic_data(num_samples=20)
    return 'data/dataset.csv'

def test_feature_engineering(sample_data):
    clf = DifficultyClassifier()
    df = pd.read_csv(sample_data)
    df_transformed = clf.add_features(df)
    assert 'flesch_score' in df_transformed.columns
    assert 'word_count' in df_transformed.columns

def test_training_flow(sample_data):
    clf = DifficultyClassifier()
    report = clf.train(sample_data)
    assert isinstance(report, str)
    assert os.path.exists('model.pkl')
    assert os.path.exists('metrics.txt')