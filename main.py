from typing import re

from data_processing import load_and_merge_data

# Veri yükleme ve birleştirme işlemlerini çağır
data = load_and_merge_data()

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
nltk.download("wordnet")


# Önişleme işlemleri için fonksiyonlar
def preprocess_text(text):
    # Küçük harfe dönüştürme
    text = text.lower()

    # Gereksiz karakterlerin kaldırılması
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Kelimelerin köklerine ayırma
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    # Stop kelimelerin kaldırılması
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


# Metin verilerinin ön işleme adımlarını uygula
data["text"] = data["text"].apply(preprocess_text)

# Metin verilerini vektörleştirme
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(data["text"])

# Etiketleri hazırlama
labels = data["label"]
