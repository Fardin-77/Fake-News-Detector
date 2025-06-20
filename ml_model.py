# FakeNewsApp/ml_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

class FakeNewsModel:
    def __init__(self):
        # Load and prepare data
        true_df = pd.read_csv('True.csv')
        fake_df = pd.read_csv('Fake.csv')

        true_df['label'] = 'REAL'
        fake_df['label'] = 'FAKE'

        data = pd.concat([true_df, fake_df], ignore_index=True)
        data = data[['text', 'label']]

        x = data['text']
        y = data['label']

        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_x = self.vectorizer.fit_transform(x)

        # Train Model
        self.model = PassiveAggressiveClassifier()
        self.model.fit(tfidf_x, y)

    def predict(self, input_text):
        text_vector = self.vectorizer.transform([input_text])
        prediction = self.model.predict(text_vector)
        return prediction[0]
