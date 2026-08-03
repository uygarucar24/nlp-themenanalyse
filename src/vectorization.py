import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Domänenspezifische Rauschwörter
EXTRA_STOPWORDS = ["deutsch", "deutsche", "deutschen", "bahn", "db", "mal", "gutefrage", "net", "nen", "hab",
                   "hallo", "einfach", "eigentlich", "umsonst", "letzter", "nächster", "normal", "sogar",
                   "montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag", "fast", "via"]

# Städte-/Regionsnamen (für die Themen-Analyse nicht relevant)
CITY_STOPWORDS = ["berlin", "berliner", "münchen", "hamburg", "köln", "kölner", "stuttgart",
                  "hannover", "leipzig", "bayern", "frankfurt", "bremen", "dresden"]

STOPWORDS = EXTRA_STOPWORDS + CITY_STOPWORDS


# Erstellung der Bag of Words Matrix
def build_bow(docs):
    vectorizer = CountVectorizer(min_df=10, max_df=0.5, stop_words=STOPWORDS)
    matrix = vectorizer.fit_transform(docs)

    return matrix, vectorizer


# Erstellung der TF-IDF Matrix (Gewichtet nach Häufigkeit und Seltenheit)
def build_tfidf(docs):
    vectorizer = TfidfVectorizer(min_df=10, max_df=0.5, stop_words=STOPWORDS)
    matrix = vectorizer.fit_transform(docs)

    return matrix, vectorizer


# Stellt die häufigsten BoW-Wörter den stärksten TF-IDF-Gewichten gegenüber
def compare_vectorizers(bow_matrix, bow_vectorizer, tfidf_matrix, tfidf_vectorizer, n_words=15):
    bow_words = bow_vectorizer.get_feature_names_out()
    tfidf_words = tfidf_vectorizer.get_feature_names_out()

    bow_sums = np.asarray(bow_matrix.sum(axis=0)).ravel()
    bow_order = bow_sums.argsort()[::-1][:n_words]

    tfidf_means = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    tfidf_order = tfidf_means.argsort()[::-1][:n_words]

    print(f"{'BoW (häufigste)':25s} TF-IDF (stärkste Gewichte)")
    print("-" * 55)
    for rank in range(n_words):
        left = bow_words[bow_order[rank]]
        right = tfidf_words[tfidf_order[rank]]
        print(f"{left:25s} {right}")
