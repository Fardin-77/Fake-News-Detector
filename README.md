# 📰 Fake News Detection Using Machine Learning

This is a simple yet powerful Fake News Detection project built using **Python, Scikit-learn, and Django** (for web version) and **Jupyter Notebook** (for demo/testing).

---

## 📌 Features

- Real-time fake/real news prediction
- Natural Language Processing (TF-IDF)
- PassiveAggressiveClassifier for efficient training
- Web-based UI using Django
- No pickle/joblib — model trains fresh or runs in memory

---

## 📁 Dataset Used

- `True.csv` – Real news articles
- `Fake.csv` – Fake news articles

Both datasets have:
- `title`
- `text`
- `subject`
- `date`

---

## 🚀 Jupyter Notebook (CLI Based)

```bash
📂 File: FakeNews_Detector.ipynb

✔ Loads both datasets
✔ Preprocesses and vectorizes using TF-IDF
✔ Trains model (PassiveAggressiveClassifier)
✔ Evaluates with accuracy & confusion matrix
✔ Accepts real-time news input for prediction (CLI)
