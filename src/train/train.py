import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from data_loader import load_data
from text_preprocessing import preprocess_text

df = load_data('train.csv')

df.drop_duplicates(inplace=True)

df['tokens'] = df['review'].apply(preprocess_text)
df['processed_review'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['processed_review'])
y = df['sentiment']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearSVC()
model.fit(X_train, y_train)

predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print(f"Validation Accuracy: {accuracy:.4f}")

if accuracy < 0.85:
    raise Exception("Model accuracy is below the required threshold of 0.85.")

models_dir = os.path.join('..', '..', 'outputs', 'models')
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, 'best_model.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("Training complete. Best model and TF-IDF vectorizer saved.")
