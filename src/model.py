import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import textstat
import joblib

class DifficultyClassifier:
    def __init__(self):
        self.model = None

    def add_features(self, df):
        """Добавляем инженерные признаки: читаемость текста."""
        df['flesch_score'] = df['question_text'].apply(textstat.flesch_reading_ease)
        df['word_count'] = df['question_text'].apply(lambda x: len(str(x).split()))
        return df

    def train(self, data_path):
        # 1. Загрузка и подготовка
        df = pd.read_csv(data_path)
        df = self.add_features(df)

        X = df[['question_text', 'avg_time_seconds', 'success_rate', 'flesch_score', 'word_count']]
        y = df['difficulty']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Пайплайн обработки
        numeric_features = ['avg_time_seconds', 'success_rate', 'flesch_score', 'word_count']
        text_features = 'question_text'

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('txt', TfidfVectorizer(), text_features)
            ])

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # 3. Обучение
        self.model.fit(X_train, y_train)

        # 4. Оценка
        print("Model Score:", self.model.score(X_test, y_test))
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)

        # 5. Сохранение артефактов
        joblib.dump(self.model, 'model.pkl')

        # ИСПРАВЛЕНИЕ: Сохраняем метрики прямо внутри метода
        with open("metrics.txt", "w") as f:
            f.write(report)

        return report

if __name__ == "__main__":
    clf = DifficultyClassifier()
    clf.train('data/dataset.csv')