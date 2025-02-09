import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_br(text):
    """Remove isolated 'br' tokens."""
    return re.sub(r'\bbr\b', '', text)

def clean_text(text):
    text = clean_html(text)
    text = remove_br(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<br\s*/?>', ' ', text)
    return text.strip()

def preprocess_text(text):
    text = clean_html(text)
    text = re.sub(r'\bbr\b', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens
