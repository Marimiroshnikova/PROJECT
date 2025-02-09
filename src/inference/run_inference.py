import os
import pickle
import pandas as pd

from data_loader import load_data
from text_preprocessing import preprocess_text

df_test = load_data('test.csv')

df_test['tokens'] = df_test['review'].apply(preprocess_text)
df_test['processed_review'] = df_test['tokens'].apply(lambda tokens: ' '.join(tokens))

models_dir = os.path.join('..', '..', 'outputs', 'models')
with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open(os.path.join(models_dir, 'best_model.pkl'), 'rb') as f:
    model = pickle.load(f)

X_test = tfidf_vectorizer.transform(df_test['processed_review'])
df_test['predicted_sentiment'] = model.predict(X_test)

predictions_dir = os.path.join('..', '..', 'outputs', 'predictions')
os.makedirs(predictions_dir, exist_ok=True)
output_path = os.path.join(predictions_dir, 'predictions.csv')
df_test.to_csv(output_path, index=False)

print("Inference complete. Predictions saved to", output_path)
