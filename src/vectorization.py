from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

EXTRA_STOPWORDS = ["deutsch", "deutsche", "deutschen", "bahn", "db"]


# Erstellung der Bag of Words Matrix
def build_bow(docs):
    vectorizer = CountVectorizer(min_df=10, max_df=0.5, stop_words=EXTRA_STOPWORDS)
    matrix = vectorizer.fit_transform(docs)

    return matrix, vectorizer


# Erstellung der TF-IDF Matrix (Gewichtet nach Häufigkeit und Seltenheit)
def build_tfidf(docs):
    vectorizer = TfidfVectorizer(min_df=10, max_df=0.5, stop_words=EXTRA_STOPWORDS)
    matrix = vectorizer.fit_transform(docs)
    
    return matrix, vectorizer