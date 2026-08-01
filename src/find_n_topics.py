import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from data_loader import load_data
from preprocessing import clean_corpus
from vectorization import build_tfidf


# Ellbogen Methode - Zeichnet den NMF-Rekonstruktionsfehler für verschiedene Werte der Themenanzahl (k)
def plot_elbow(tfidf_matrix, k_min=2, k_max=10):
    k_values = []
    errors = []
    for k in range(k_min, k_max + 1):
        model = NMF(n_components=k, init="nndsvda", beta_loss="kullback-leibler",
                    solver="mu", max_iter=500, random_state=24)
        model.fit(tfidf_matrix)
        k_values.append(k)
        errors.append(model.reconstruction_err_)

    plt.figure()
    plt.plot(k_values, errors, marker="o")
    plt.xlabel("Anzahl Themen (k)")
    plt.ylabel("Rekonstruktionsfehler (kleiner = besser)")
    plt.title("Ellbogen-Methode (NMF)")
    plt.savefig("results/ellbogen.png")
    plt.show()


# Berechnet und zeichnet den Kohärenzwert (C_v) bei unterschiedlicher Themenanzahl
def plot_coherence(docs, tfidf_matrix, vectorizer, k_min=2, k_max=10):
    words = vectorizer.get_feature_names_out()

    texts_tokens = []
    for d in docs:
        texts_tokens.append(d.split())
    dictionary = Dictionary(texts_tokens)

    k_values = []
    scores = []
    for k in range(k_min, k_max + 1):
        model = NMF(n_components=k, init="nndsvda", beta_loss="kullback-leibler",
                    solver="mu", max_iter=500, random_state=24)
        model.fit(tfidf_matrix)

        topics = []
        for topic in model.components_:
            top_words = []
            for i in topic.argsort()[::-1][:10]:
                top_words.append(words[i])
            topics.append(top_words)

        # processes=1 wegen gensim unter Windows
        cm = CoherenceModel(topics=topics, texts=texts_tokens,
                            dictionary=dictionary, coherence="c_v", processes=1)
        score = cm.get_coherence()
        k_values.append(k)
        scores.append(score)
        print(f"k={k}: Coherence={score:.4f}")

    plt.figure()
    plt.plot(k_values, scores, marker="o", color="#C44E52")
    plt.xlabel("Anzahl Themen (k)")
    plt.ylabel("Coherence Score (c_v, höher = besser)")
    plt.title("Coherence-Score-Methode (NMF)")
    plt.savefig("results/coherence.png")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    docs = clean_corpus(df["text"])
    tfidf_matrix, tfidf_vectorizer = build_tfidf(docs)
    plot_elbow(tfidf_matrix)
    plot_coherence(docs, tfidf_matrix, tfidf_vectorizer)
