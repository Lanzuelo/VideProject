import pandas as pd
import numpy as np
import random
import os  # <--- Важный импорт

def generate_synthetic_data(num_samples=200):
    """Генерирует датасет с вопросами и метриками."""
    data = []

    topics = ['Math', 'History', 'Science', 'Programming']
    difficulties = ['Easy', 'Medium', 'Hard']

    for _ in range(num_samples):
        difficulty = np.random.choice(difficulties, p=[0.4, 0.4, 0.2])
        topic = random.choice(topics)

        # Логика генерации признаков в зависимости от сложности
        if difficulty == 'Easy':
            text_len = random.randint(5, 15)
            avg_time = random.uniform(10, 30)
            success_rate = random.uniform(0.75, 1.0)
            text = f"Simple {topic} question " * (text_len // 3)
        elif difficulty == 'Medium':
            text_len = random.randint(15, 30)
            avg_time = random.uniform(30, 90)
            success_rate = random.uniform(0.40, 0.75)
            text = f"Standard level {topic} question regarding specific details " * (text_len // 6)
        else: # Hard
            text_len = random.randint(30, 60)
            avg_time = random.uniform(90, 300)
            success_rate = random.uniform(0.0, 0.40)
            text = f"Complex advanced {topic} problem requiring deep analysis and synthesis " * (text_len // 8)

        data.append({
            'question_text': text,
            'topic': topic,
            'avg_time_seconds': round(avg_time, 1),
            'success_rate': round(success_rate, 2),
            'difficulty': difficulty
        })

    df = pd.DataFrame(data)

    # --- ИСПРАВЛЕНИЕ: Создаем папку data, если её нет ---
    if not os.path.exists('data'):
        os.makedirs('data')
    # ----------------------------------------------------

    df.to_csv('data/dataset.csv', index=False)
    print("✅ Dataset generated at data/dataset.csv")

if __name__ == "__main__":
    generate_synthetic_data()